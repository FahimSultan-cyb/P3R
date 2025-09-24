import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_dataset import create_dataloader
from src.models.metrics import calculate_comprehensive_metrics, print_evaluation_results
from configs.config import P3RConfig

# class P3RTrainer:
#     def __init__(self, model=None, config=None):
#         if config is None:
#             config = P3RConfig()
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         if model is None:
#             model = P3RHeadGateModel(config=config)

#         for submodule in model.children():
#             submodule.to(self.device)
#         self.model = model

#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.AdamW(
#             [p for p in self.model.parameters() if p.requires_grad], 
#             lr=config.learning_rate
#         )
#     def train(self, train_csv, epochs=None):
#         if epochs is None:
#             epochs = self.config.num_epochs
            
#         train_loader = create_dataloader(train_csv, self.model.tokenizer, self.config, shuffle=True)
        
#         print(f"Training samples: {len(train_loader.dataset)}")
#         trainable_params, total_params = self.model.count_parameters()
#         print(f"Trainable parameters: {trainable_params:,}")
#         print(f"Total parameters: {total_params:,}")
#         print(f"Frozen parameters: {total_params - trainable_params:,}")
        
#         self.model.train()
        
#         for epoch in range(epochs):
#             total_loss = 0
#             correct = 0
#             total = 0
            
#             progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
#             for batch in progress_bar:
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 self.optimizer.zero_grad()
#                 logits = self.model(chunks, full_code, attention_mask)
#                 loss = self.criterion(logits, labels)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
                
#                 progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
#             train_acc = 100. * correct / total
#             print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        
#         return self.model
    
#     def evaluate(self, test_csv, dataset_name="Test"):
#         test_loader = create_dataloader(test_csv, self.model.tokenizer, self.config, shuffle=False)
#         print(f"Test samples for {dataset_name}: {len(test_loader.dataset)}")
        
#         self.model.eval()
#         all_preds = []
#         all_labels = []
#         all_probs = []
        
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f'Evaluating {dataset_name}'):
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 logits = self.model(chunks, full_code, attention_mask)
#                 probs = torch.softmax(logits, dim=-1)
#                 _, predicted = torch.max(logits, 1)
                
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(probs[:, 1].cpu().numpy())
        
#         metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
#         print_evaluation_results(metrics, dataset_name)
        
#         self.model.train()
#         return metrics
    
#     def evaluate_multiple_datasets(self, test_configs):
#         print("\n" + "="*80)
#         print("MULTI-DATASET EVALUATION RESULTS")
#         print("="*80)
        
#         all_results = {}
        
#         for test_config in test_configs:
#             print(f"\nLoading test dataset: {test_config['name']}...")
            
#             temp_config = P3RConfig()
#             temp_config.code_col = test_config.get('code_col', 'func')
#             temp_config.label_col = test_config.get('label_col', 'label')
            
#             test_loader = create_dataloader(test_config['csv'], self.model.tokenizer, temp_config, shuffle=False)
#             print(f"Test samples for {test_config['name']}: {len(test_loader.dataset)}")
            
#             self.model.eval()
#             all_preds = []
#             all_labels = []
#             all_probs = []
            
#             with torch.no_grad():
#                 for batch in tqdm(test_loader, desc=f'Evaluating {test_config["name"]}'):
#                     chunks = batch['chunks'].to(self.device)
#                     full_code = batch['full_code'].to(self.device)
#                     attention_mask = batch['attention_mask'].to(self.device)
#                     labels = batch['label'].to(self.device)
                    
#                     logits = self.model(chunks, full_code, attention_mask)
#                     probs = torch.softmax(logits, dim=-1)
#                     _, predicted = torch.max(logits, 1)
                    
#                     all_preds.extend(predicted.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
#                     all_probs.extend(probs[:, 1].cpu().numpy())
            
#             metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
#             print_evaluation_results(metrics, test_config['name'])
#             all_results[test_config['name']] = metrics
        
#         print("\n" + "="*80)
#         print("SUMMARY OF ALL RESULTS")
#         print("="*80)
        
#         print(f"{'Dataset':<20} {'Accuracy':<10} {'Bal_Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
#         print("-" * 80)
        
#         for dataset_name, metrics in all_results.items():
#             print(f"{dataset_name:<20} {metrics['acc']:<10.4f} {metrics['bal_acc']:<10.4f} "
#                   f"{metrics['prec']:<10.4f} {metrics['rec']:<10.4f} {metrics['f1']:<10.4f} {metrics['mcc']:<10.4f}")
        
#         trainable_params, total_params = self.model.count_parameters()
#         frozen_params = total_params - trainable_params
#         print(f"\nModel Comparison:")
#         print(f"Frozen model parameters: {frozen_params:,}")
#         print(f"P³R + HeadGate trainable parameters: {trainable_params:,}")
#         print(f"Ratio: {trainable_params/frozen_params:.1%} of original model size")
        
#         self.model.train()
#         return all_results




class P3RTrainer:
    def __init__(self, model=None, config=None):
        if config is None:
            config = P3RConfig()
        self.config = config
        self.device = torch.device(config.device)
        
        if model is None:
            self.model = P3RHeadGateModel(config=config).to(self.device)
        else:
            self.model = model.to(self.device)
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad], 
                                   lr=config.learning_rate)
        
    def train(self, train_csv, epochs=None):
        if epochs is None:
            epochs = self.config.num_epochs
            
        train_loader = create_dataloader(train_csv, self.model.tokenizer, self.config, shuffle=True)
        
        print(f"Training samples: {len(train_loader.dataset)}")
        trainable_params, total_params = self.model.count_parameters()
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(chunks, full_code, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
            train_acc = 100. * correct / total
            print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        
        return self.model
    
    def evaluate(self, test_csv, dataset_name="Test"):
        test_loader = create_dataloader(test_csv, self.model.tokenizer, self.config, shuffle=False)
        print(f"Test samples for {dataset_name}: {len(test_loader.dataset)}")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Evaluating {dataset_name}'):
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(chunks, full_code, attention_mask)
                probs = torch.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
        print_evaluation_results(metrics, dataset_name)
        
        self.model.train()
        return metrics
    
    def evaluate_multiple_datasets(self, test_configs):
        print("\n" + "="*80)
        print("MULTI-DATASET EVALUATION RESULTS")
        print("="*80)
        
        all_results = {}
        
        for test_config in test_configs:
            print(f"\nLoading test dataset: {test_config['name']}...")
            
            temp_config = P3RConfig()
            temp_config.code_col = test_config.get('code_col', 'func')
            temp_config.label_col = test_config.get('label_col', 'label')
            
            test_loader = create_dataloader(test_config['csv'], self.model.tokenizer, temp_config, shuffle=False)
            print(f"Test samples for {test_config['name']}: {len(test_loader.dataset)}")
            
            self.model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Evaluating {test_config["name"]}'):
                    chunks = batch['chunks'].to(self.device)
                    full_code = batch['full_code'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = self.model(chunks, full_code, attention_mask)
                    probs = torch.softmax(logits, dim=-1)
                    _, predicted = torch.max(logits, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
            print_evaluation_results(metrics, test_config['name'])
            all_results[test_config['name']] = metrics
        
        print("\n" + "="*80)
        print("SUMMARY OF ALL RESULTS")
        print("="*80)
        
        print(f"{'Dataset':<20} {'Accuracy':<10} {'Bal_Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
        print("-" * 80)
        
        for dataset_name, metrics in all_results.items():
            print(f"{dataset_name:<20} {metrics['acc']:<10.4f} {metrics['bal_acc']:<10.4f} "
                  f"{metrics['prec']:<10.4f} {metrics['rec']:<10.4f} {metrics['f1']:<10.4f} {metrics['mcc']:<10.4f}")
        
        trainable_params, total_params = self.model.count_parameters()
        frozen_params = total_params - trainable_params
        print(f"\nModel Comparison:")
        print(f"Frozen model parameters: {frozen_params:,}")
        print(f"P³R + HeadGate trainable parameters: {trainable_params:,}")
        print(f"Ratio: {trainable_params/frozen_params:.1%} of original model size")
        
        self.model.train()
        return all_results
