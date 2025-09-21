import re
import pandas as pd
from typing import List

class NeurosymbolicFeatureExtractor:
    def __init__(self):
        self.c_types = {'int32', 'uint32', 'int', 'uint', 'char', 'float', 'double', 'void', 'bool', 'size_t'}
        self.buffer_functions = {'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'memcpy', 'memmove', 'strncpy', 'strncat'}
        self.memory_functions = {'malloc', 'calloc', 'realloc', 'free', 'alloca'}
        self.io_functions = {'printf', 'fprintf', 'fopen', 'fclose', 'fread', 'fwrite', 'read', 'write'}
        self.sync_functions = {'pthread_mutex_lock', 'pthread_mutex_unlock', 'sem_wait', 'sem_post'}
        self.cfe_functions = {'CFE_SB_RcvMsg', 'CFE_SB_SendMsg', 'CFE_ES_RunLoop', 'CFE_SB_CreatePipe'}
        
    def clean_code(self, code: str) -> str:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)
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
        
        return features

    def extract_vulnerability_patterns(self, code: str) -> List[str]:
        features = []
        if 'CFE_SB_PEND_FOREVER' in code:
            features.append('vuln_pattern: infinite_wait')
        if re.search(r'while\s*\([^)]*TRUE[^)]*\)', code):
            features.append('vuln_pattern: infinite_loop')
        if 'strcpy' in code or 'strcat' in code:
            features.append('vuln_pattern: buffer_overflow_risk')
        if 'malloc' in code and 'free' not in code:
            features.append('vuln_pattern: potential_memory_leak')
        return features

    def _extract_args(self, call: str) -> str:
        args_match = re.search(r'\(([^)]*)\)', call)
        if args_match:
            args = args_match.group(1).strip()
            arg_count = len([arg for arg in args.split(',') if arg.strip()]) if args else 0
            return f'argc_{arg_count}'
        return 'argc_0'

    def extract_all_features(self, code: str) -> List[str]:
        cleaned_code = self.clean_code(code)
        all_features = []
        all_features.extend(self.extract_declarations(cleaned_code))
        all_features.extend(self.extract_assignments(cleaned_code))
        all_features.extend(self.extract_function_calls(cleaned_code))
        all_features.extend(self.extract_control_flow(cleaned_code))
        all_features.extend(self.extract_vulnerability_patterns(cleaned_code))
        return list(set(all_features))

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    extractor = NeurosymbolicFeatureExtractor()
    
    neurosymbolic_features = []
    for _, row in df.iterrows():
        code = str(row['func'])
        features = extractor.extract_all_features(code)
        neurosymbolic_features.append(str(features))
    
    df['neuro'] = neurosymbolic_features
    df.to_csv(output_path, index=False)
    return df
