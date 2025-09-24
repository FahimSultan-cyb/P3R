from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_trainer import P3RTrainer
from configs import P3RConfig
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Initializing P³R + HeadGate Model...")
    
    config = P3RConfig(
        model_name="microsoft/unixcoder-base",
        num_epochs=5,
        batch_size=4,
        learning_rate=1e-4,
        code_col='func',
        label_col='target'
    )
    
    model = P3RHeadGateModel(config=config)
    trainer = P3RTrainer(model=model, config=config)
    
    print(f"Using device: {config.device}")
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Ratio: {trainable_params/(total_params - trainable_params):.1%} of frozen model size")
    
    print("\nStarting training...")
    trainer.train("train.csv")
    
    print("\nEvaluating on test set...")
    results = trainer.evaluate("test.csv", "Test_Set")
    
    test_configs = [
        {
            'csv': 'big_vuln.csv',
            'label_col': 'label',
            'code_col': 'func',
            'name': 'BIG_VULTEST'
        },
        {
            'csv': 'mix_vuln.csv',
            'label_col': 'label',
            'code_col': 'func',
            'name': 'MIX_VULTEST'
        }
    ]
    
    if len(test_configs) > 0:
        print("\nEvaluating on multiple datasets...")
        all_results = trainer.evaluate_multiple_datasets(test_configs)

if __name__ == "__main__":
    main()
