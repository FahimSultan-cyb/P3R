from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_trainer import P3RTrainer
from configs import P3RConfig

def basic_example():
    config = P3RConfig(
        model_name="microsoft/unixcoder-base",
        num_epochs=5,
        batch_size=4,
        learning_rate=1e-4
    )
    
    model = P3RHeadGateModel(config=config)
    trainer = P3RTrainer(model=model, config=config)
    
    trainer.train("train.csv")
    results = trainer.evaluate("test.csv")
    
    return results

def multi_model_example():
    models = [
        "microsoft/unixcoder-base",
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base"
    ]
    
    for model_name in models:
        print(f"Training with {model_name}")
        config = P3RConfig(model_name=model_name)
        model = P3RHeadGateModel(config=config)
        trainer = P3RTrainer(model=model, config=config)
        
        trainer.train("train.csv")
        results = trainer.evaluate("test.csv", f"{model_name}_results")

if __name__ == "__main__":
    basic_example()
