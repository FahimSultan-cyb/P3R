from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_trainer import P3RTrainer
from configs.config import P3RConfig

def custom_model_example():
    config = P3RConfig(
        model_name="Salesforce/codet5-base",
        num_prompts=8,
        prompt_length=12,
        num_classes=2,
        max_length=1024,
        chunk_size=256,
        stride=128,
        batch_size=2,
        learning_rate=5e-5,
        num_epochs=10,
        dropout=0.2,
        code_col='code_snippet',
        label_col='vulnerability'
    )
    
    model = P3RHeadGateModel(config=config)
    trainer = P3RTrainer(model=model, config=config)
    
    print("Custom configuration loaded:")
    print(f"Model: {config.model_name}")
    print(f"Prompts: {config.num_prompts}")
    print(f"Prompt length: {config.prompt_length}")
    print(f"Max length: {config.max_length}")
    print(f"Chunk size: {config.chunk_size}")
    
    trainer.train("custom_train.csv")
    results = trainer.evaluate("custom_test.csv", "Custom_Dataset")
    
    return results

def roberta_based_example():
    config = P3RConfig(
        model_name="microsoft/codebert-base",
        num_prompts=6,
        prompt_length=10,
        batch_size=8,
        learning_rate=2e-4
    )
    
    model = P3RHeadGateModel(config=config)
    trainer = P3RTrainer(model=model, config=config)
    
    trainer.train("roberta_train.csv")
    results = trainer.evaluate("roberta_test.csv", "RoBERTa_Results")
    
    return results

if __name__ == "__main__":
    custom_model_example()
    roberta_based_example()
