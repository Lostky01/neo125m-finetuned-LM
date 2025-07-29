import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize with truncation and padding
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token 

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# This is your tokenized dataset (plural)
tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")


training_args = TrainingArguments(
    output_dir="./neo-125m-finetuned",
    per_device_train_batch_size=1,  
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer
)

# TRAIN THAT BEAST
trainer.train()

# SAVE THIS MONSTROSITY
model.save_pretrained("./neo-125m-finetuned")
tokenizer.save_pretrained("./neo-125m-finetuned")
