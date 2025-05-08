import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    # Load tokenizer & model
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with chosen model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("json", data_files={"train": f"{args.train_data}/data.jsonl"})
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], text_target=examples["response"], truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        save_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("/opt/ml/model")

if __name__ == "__main__":
    main()
