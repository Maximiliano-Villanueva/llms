# -*- coding: utf-8 -*-
"""
Fine-tuning Gemma 3 1B IT with PEFT LoRA for Causal LM (using transformers.Trainer)

This notebook demonstrates how to fine-tune the google/gemma-3-1b-it model
on a custom dataset (CSV format) for a question-answering task using
Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA, and the standard
transformers.Trainer.
"""

# @title 1. Install Dependencies
# !pip install -q -U transformers datasets accelerate peft bitsandbytes huggingface_hub pandas torch

import os
import pandas as pd
import torch
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    Trainer, # Use standard Trainer
    DataCollatorForLanguageModeling # Use standard data collator
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import gc # Garbage Collection

gemma_dir = os.path.join(os.getcwd(), "gemma")
os.chdir(gemma_dir)


# @title 2. Login to Hugging Face Hub
# Optional: Login to Hugging Face Hub.
# from google.colab import userdata
# hf_token = userdata.get('HF_TOKEN')
hf_token = os.getenv("HF_TOKEN")  # Use the token you provided
login(token=hf_token)

# @title 3. Configuration
model_id = "google/gemma-3-1b-it"
data_path = "data/cars.csv" # Path to your CSV file
output_dir = "./gemma3-1b-cars-finetuned-trainer" # Directory to save the fine-tuned model adapter
lora_rank = 128
lora_alpha = 128
lora_dropout = 0.05

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training Arguments
num_train_epochs = 20
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
optim = "paged_adamw_8bit"
save_steps = 50
logging_steps = 2
max_seq_length = 512 # Maximum sequence length for tokenization
device_map = {"": 0}
# device_map = "auto"

# @title 4. Create Dummy Data (if needed)
# Uncomment and run this cell to create a dummy data/cars.csv file if needed.

# print(f"Creating dummy data at {data_path}...")
# os.makedirs(os.path.dirname(data_path), exist_ok=True)
# dummy_data = {
#     'input': [
#         "What is the reference for the door on an Audi A3?",
#         "What's the part number for BMW 3 Series headlight?",
#         "Need the code for a Ford Focus wing mirror.",
#         "Part number for Mercedes C-Class brake pads?",
#         "Reference for VW Golf radiator fan?"
#     ],
#     'label': [
#         "AA34517",
#         "BM45982",
#         "FFM6789",
#         "MCC9876",
#         "VWG0101"
#     ]
# }
# df = pd.DataFrame(dummy_data)
# df.to_csv(data_path, index=False)
# print("Dummy data created.")

# @title 5. Load Dataset
print(f"Loading dataset from {data_path}...")
# dataset = load_dataset('csv', data_files=data_path, split='train').select(range(5))  # Limit to first 5 samples
dataset = load_dataset('csv', data_files=data_path, split='train')
print(dataset.features)
# Optional: Split dataset
# dataset = dataset.train_test_split(test_size=0.1)
# train_dataset = dataset["train"]
# eval_dataset = dataset["test"]

train_dataset = dataset # Using full dataset for training
print(f"Dataset loaded. Size: {len(train_dataset)}")
print("Sample entry:", train_dataset[0])

# @title 6. Load Tokenizer and Prepare Data Formatting/Tokenization Function

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set padding token and side for Causal LM
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for Causal LM

print("Defining formatting and tokenization function...")
def format_and_tokenize(row):
    # --- 1. Format the prompt using the chat template ---
    user_input = str(row['input']) if row['input'] is not None else ""
    model_label = str(row['label']) if row['label'] is not None else ""

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input},
         {"role": "model", "content": model_label}],
        tokenize=False, # We'll tokenize in the next step
        add_generation_prompt=False # Include the full conversation
    )
    # Add EOS token at the end for Causal LM training
    formatted_prompt += tokenizer.eos_token

    # --- 2. Tokenize the formatted prompt ---
    # The result will contain `input_ids`, `attention_mask`.
    # The `Trainer` with `DataCollatorForLanguageModeling` will handle labels.
    result = tokenizer(
        formatted_prompt,
        truncation=True,
        max_length=max_seq_length,
        padding=False, # Pad later dynamically with data collator
        return_tensors=None, # Return lists of IDs
    )
    # # Optional: Manually create labels (usually handled by DataCollatorForLanguageModeling)
    # result["labels"] = result["input_ids"].copy()

    return result

# Apply the function to the dataset
print("Applying formatting and tokenization...")
tokenized_train_dataset = train_dataset.map(
    format_and_tokenize,
    remove_columns=list(train_dataset.features) # Remove original columns
)

# Optional: Apply to eval dataset if you have one
# tokenized_eval_dataset = eval_dataset.map(
#     format_and_tokenize,
#     remove_columns=list(eval_dataset.features)
# )

print("Sample tokenized entry keys:", tokenized_train_dataset[0].keys())
# print("Decoded sample input_ids:", tokenizer.decode(tokenized_train_dataset[0]['input_ids']))

# @title 7. Load Model, Configure Quantization, and Apply PEFT

# Configure BitsAndBytes
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

print(f"Loading base model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config if use_4bit else None,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation='eager'
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Prepare model for k-bit training
if use_4bit:
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Apply PEFT to the model - THIS IS DONE EXPLICITLY HERE
print("Applying PEFT LoRA configuration to the model...")
model = get_peft_model(model, peft_config)
print("PEFT model created.")
model.print_trainable_parameters() # Show trainable parameters

# @title 8. Define Data Collator
# Data collator handles dynamic padding and creates labels for Causal LM
# mlm=False indicates Causal Language Modeling (not Masked Language Modeling)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("Data collator for Causal LM initialized.")

# @title 9. Define Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=0.001,
    fp16=False, # Set True if using float16
    bf16=True, # Set True if GPU supports bfloat16
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    # evaluation_strategy="steps", # Uncomment if using eval_dataset
    # eval_steps=50, # Uncomment if using eval_dataset
)

# @title 10. Instantiate Trainer (using standard transformers.Trainer)
trainer = Trainer(
    model=model,                       # The PEFT model
    args=training_arguments,           # Training arguments
    train_dataset=tokenized_train_dataset, # Tokenized training dataset
    # eval_dataset=tokenized_eval_dataset, # Tokenized evaluation dataset (optional)
    tokenizer=tokenizer,               # Tokenizer (needed for saving)
    data_collator=data_collator,       # Data collator for dynamic padding & labels
)
print("Standard Trainer initialized.")

# @title 11. Start Training
print("Starting training...")
train_result = trainer.train()

# Log & save metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
print("Training finished.")

# @title 12. Save the Fine-Tuned Adapter
print(f"Saving LoRA adapter weights to {output_dir}...")
# Saves the adapter config and weights (only the trainable parts)
trainer.save_model()
# The base model is NOT saved here, only the adapter.
print("Adapter saved.")

# Optional: Save the tokenizer if you made changes
# tokenizer.save_pretrained(output_dir)

# @title 13. Clean Up GPU Memory
del model
del trainer
# del tokenized_train_dataset # Free up dataset memory if needed
# del data_collator
gc.collect()
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up training objects and CUDA cache.")

# === INFERENCE ===
# (Inference steps remain the same as the previous notebook)

# @title 14. Load Base Model and Fine-tuned Adapter for Inference
print("Loading base model for inference...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config if use_4bit else None,
    device_map=device_map,
    trust_remote_code=True,
)
print("Base model loaded.")

print(f"Loading PEFT adapter from {output_dir}...")
model_for_inference = PeftModel.from_pretrained(base_model, output_dir)
print("PEFT adapter loaded.")

model_for_inference.eval()
print("Model set to evaluation mode.")

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# @title 15. Run Inference with Fine-tuned Model

pipe = pipeline(
    task="text-generation",
    model=model_for_inference,
    tokenizer=tokenizer,
    max_new_tokens=50
)

test_prompt = "What is the reference for the door on an Audi A3?"
# test_prompt = "Part number for Mercedes C-Class brake pads?"

messages = [{"role": "user", "content": test_prompt}]
prompt_for_model = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n--- Inference ---")
print(f"Formatted Prompt:\n{prompt_for_model}")

result = pipe(prompt_for_model)
full_output = result[0]['generated_text']
print(f"\nFull Output:\n{full_output}")

response_part = full_output.split("<start_of_turn>model")[-1]
response_part = response_part.strip().replace("<end_of_turn>", "").strip()
print(f"\nGenerated Response (extracted): {response_part}")


# Example 2
test_prompt_2 = "What's the part number for BMW 3 Series headlight?"
messages_2 = [{"role": "user", "content": test_prompt_2}]
prompt_for_model_2 = tokenizer.apply_chat_template(
    messages_2, tokenize=False, add_generation_prompt=True
)
print(f"\nFormatted Prompt 2:\n{prompt_for_model_2}")
result_2 = pipe(prompt_for_model_2)
full_output_2 = result_2[0]['generated_text']
print(f"\nFull Output 2:\n{full_output_2}")
response_part_2 = full_output_2.split("<start_of_turn>model")[-1].strip().replace("<end_of_turn>", "").strip()
print(f"\nGenerated Response 2 (extracted): {response_part_2}")

print("\n--- End of Script ---")

# Optional: Merge adapter weights
print("Merging adapter weights...")
merged_model = model_for_inference.merge_and_unload()
print("Adapter merged.")
merged_model_dir = f"{output_dir}-merged"
merged_model.save_pretrained(merged_model_dir)
tokenizer.save_pretrained(merged_model_dir)
print(f"Full merged model saved to {merged_model_dir}")