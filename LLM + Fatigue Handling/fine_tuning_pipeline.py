import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List
import numpy as np
import input_process
from model_wrapper_with_mlp_adapter import FeaturePrefixAdapter, PrefixLLaMAModel

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
FEATURE_DIM = 12
EMBEDDING_DIM = 4096
PREFIX_TOKEN_COUNT = 5
MAX_LENGTH = 256
BATCH_SIZE = 2


def train():
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load LLaMA model and apply LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    llama_model = get_peft_model(base_model, lora_config)

    # 3. Initialize MLP Adapter and wrap with Prefix model
    adapter = FeaturePrefixAdapter()
    full_model = PrefixLLaMAModel(llama_model, adapter)

    # 4. Create toy dataset (replace with real data)
    #features = np.random.rand(20, FEATURE_DIM).tolist()
    #responses = ["Suggest the driver to take a short rest." for _ in range(20)]

    dataset = input_process.SensorTextDataset(features, responses, tokenizer)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./llama_prefix_finetune",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none",
        save_total_limit=1,
        bf16=True,  # if GPU supports it
    )

    # 6. Hugging Face Trainer
    trainer = Trainer(
        model=full_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=input_process.custom_collate
    )

    # 7. Start training
    trainer.train()

    # 8. Save everything
    trainer.save_model("./llama_prefix_final_model")
    tokenizer.save_pretrained("./llama_prefix_final_model")
    torch.save(adapter.state_dict(), "./llama_prefix_final_model/prefix_adapter.pth")
