import re
import pandas as pd
from typing import List
from collections import defaultdict

class NeurosymbolicFeatureExtractor:
    def __init__(self):
        self.c_types = {'int32', 'uint32', 'int', 'uint', 'char', 'float', 'double', 'void', 'bool', 'size_t'}
        self.buffer_functions = {'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'memcpy', 'memmove', 'strncpy', 'strncat'}
        self.memory_functions = {'malloc', 'calloc', 'realloc', 'free', 'alloca'}
        self.io_functions = {'printf', 'fprintf', 'fopen', 'fclose', 'fread', 'fwrite', 'read', 'write'}
        self.sync_functions = {'pthread_mutex_lock', 'pthread_mutex_unlock', 'sem_wait', 'sem_post'}
        self.cfe_functions = {'CFE_SB_RcvMsg', 'CFE_SB_SendMsg', 'CFE_ES_RunLoop', 'CFE_SB_CreatePipe'}
        
    def clean_code(self, code: str) -> str:
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'[ \t]+', ' ', code)
        return code.strip()

    def extract_declarations(self, code: str) -> List[str]:
        features = []
        decl_pattern = r'(\w+(?:\s*\*)*)\s+(\w+)(?:\s*=\s*([^;,]+))?[;,]'
        for match in re.finditer(decl_pattern, code):
            var_type, var_name, init_value = match.groups()
            if var_type in self.c_types or var_type.endswith('_t'):
                if init_value:
                    features.append(f'declare_init {var_type} {var_name} = {init_value.strip()}')
                else:
                    features.append(f'declare {var_type} {var_name}')
        return features

    def extract_assignments(self, code: str) -> List[str]:
        features = []
        assign_pattern = r'(\w+(?:\.\w+|\[\w*\])*)\s*=\s*([^;]+);'
        for match in re.finditer(assign_pattern, code):
            lvalue, rvalue = match.groups()
            rvalue = rvalue.strip()
            if '(' in rvalue and ')' in rvalue:
                features.append(f'assign_call: {lvalue} = {rvalue}')
            elif any(op in rvalue for op in ['&', '*', '->', '.']):
                features.append(f'assign_ptr: {lvalue} = {rvalue}')
            else:
                features.append(f'assign: {lvalue} = {rvalue}')
        return features

    def extract_function_calls(self, code: str) -> List[str]:
        features = []
        call_pattern = r'(\w+)\s*\([^)]*\)'
        for match in re.finditer(call_pattern, code):
            func_name = match.group(1)
            full_call = match.group(0)
            if func_name in self.buffer_functions:
                features.append(f'call_buffer_risk: {func_name}({self._extract_args(full_call)})')
            elif func_name in self.memory_functions:
                features.append(f'call_memory: {func_name}({self._extract_args(full_call)})')
            elif func_name in self.cfe_functions:
                features.append(f'call_cfe: {func_name}({self._extract_args(full_call)})')
            elif func_name.startswith('CFE_') or func_name.startswith('OS_'):
                features.append(f'call_system: {func_name}({self._extract_args(full_call)})')
            else:
                features.append(f'call: {func_name}({self._extract_args(full_call)})')
        return features

    def extract_control_flow(self, code: str) -> List[str]:
        features = []
        if_pattern = r'if\s*\([^)]+\)'
        for match in re.finditer(if_pattern, code):
            condition = match.group(0)
            if '==' in condition:
                features.append(f'if_equality: {condition}')
            elif '!=' in condition:
                features.append(f'if_inequality: {condition}')
            elif any(op in condition for op in ['<', '>', '<=', '>=']):
                features.append(f'if_comparison: {condition}')
            else:
                features.append(f'if: {condition}')
        
        while_pattern = r'while\s*\([^)]+\)'
        for match in re.finditer(while_pattern, code):
            features.append(f'while: {match.group(0)}')
        
        for_pattern = r'for\s*\([^)]+\)'
        for match in re.finditer(for_pattern, code):
            features.append(f'for: {match.group(0)}')
        
        switch_pattern = r'switch\s*\([^)]+\)'
        for match in re.finditer(switch_pattern, code):
            features.append(f'switch: {match.group(0)}')
        return features

    def extract_pointer_operations(self, code: str) -> List[str]:
        features = []
        deref_pattern = r'\*(\w+(?:\.\w+)*)'
        for match in re.finditer(deref_pattern, code):
            features.append(f'deref: *{match.group(1)}')
        
        addr_pattern = r'&(\w+(?:\.\w+)*(?:\[\w*\])*)'
        for match in re.finditer(addr_pattern, code):
            features.append(f'addr_of: &{match.group(1)}')
        
        arrow_pattern = r'(\w+)->(\w+)'
        for match in re.finditer(arrow_pattern, code):
            features.append(f'ptr_access: {match.group(1)}->{match.group(2)}')
        return features

    def extract_array_operations(self, code: str) -> List[str]:
        features = []
        array_pattern = r'(\w+)\[([^\]]*)\]'
        for match in re.finditer(array_pattern, code):
            array_name, index = match.groups()
            if index.isdigit():
                features.append(f'array_const_idx: {array_name}[{index}]')
            elif index == '':
                features.append(f'array_empty_idx: {array_name}[]')
            else:
                features.append(f'array_var_idx: {array_name}[{index}]')
        return features

    def extract_vulnerability_patterns(self, code: str) -> List[str]:
        features = []
        if 'CFE_SB_PEND_FOREVER' in code:
            features.append('vuln_pattern: infinite_wait')
        if re.search(r'while\s*\([^)]*TRUE[^)]*\)', code):
            features.append('vuln_pattern: infinite_loop')
        if 'strcpy' in code or 'strcat' in code:
            features.append('vuln_pattern: buffer_overflow_risk')
        if re.search(r'malloc.*free', code, re.DOTALL):
            features.append('vuln_pattern: memory_lifecycle')
        elif 'malloc' in code and 'free' not in code:
            features.append('vuln_pattern: potential_memory_leak')
        if re.search(r'if\s*\([^)]*status[^)]*\)', code):
            features.append('vuln_pattern: status_check_present')
        else:
            features.append('vuln_pattern: missing_error_check')
        return features

    def _extract_args(self, call: str) -> str:
        args_match = re.search(r'\(([^)]*)\)', call)
        if args_match:
            args = args_match.group(1).strip()
            arg_count = len([arg for arg in args.split(',') if arg.strip()]) if args else 0
            return f'argc_{arg_count}'
        return 'argc_0'

    def extract_all_features(self, code: str) -> List[str]:
        all_features = []
        all_features.extend(self.extract_declarations(code))
        all_features.extend(self.extract_assignments(code))
        all_features.extend(self.extract_function_calls(code))
        all_features.extend(self.extract_control_flow(code))
        all_features.extend(self.extract_pointer_operations(code))
        all_features.extend(self.extract_array_operations(code))
        all_features.extend(self.extract_vulnerability_patterns(code))
        return list(set(all_features))

    def preprocess_dataset(input_csv: str, output_csv: str) -> pd.DataFrame:
        extractor = NeurosymbolicFeatureExtractor()
        df = pd.read_csv(input_csv)
    
        print(f"Processing {len(df)} samples...")
    
        df['func'] = df['func'].apply(extractor.clean_code)
    
        neurosymbolic_features = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} samples")
        
            code = row['func']
            features = extractor.extract_all_features(code)
            neurosymbolic_features.append(str(features))
    
        df['neuro'] = neurosymbolic_features
        df.to_csv(output_csv, index=False)
    
        print(f"Preprocessing completed. Output saved to: {output_csv}")
        return df

    def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        extractor = NeurosymbolicFeatureExtractor()

        print(f"Processing {len(df)} samples...")

        df['func'] = df['func'].apply(extractor.clean_code)

        neurosymbolic_features = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} samples")

            code = row['func']
            features = extractor.extract_all_features(code)
            neurosymbolic_features.append(str(features))

        df['neuro'] = neurosymbolic_features
        return df




