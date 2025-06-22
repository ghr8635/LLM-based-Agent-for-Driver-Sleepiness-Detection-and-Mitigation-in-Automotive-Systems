import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import input_process  # Must contain load_csv_dataset, SensorTextDataset, custom_collate
from model_wrapper_with_mlp_adapter import FeaturePrefixAdapter, PrefixLLaMAModel
import random
import numpy as np
import os

# === Configuration ===
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
FEATURE_DIM = 9  # Updated based on new feature list
EMBEDDING_DIM = 4096
PREFIX_TOKEN_COUNT = 5
MAX_LENGTH = 256
BATCH_SIZE = 2

# Set up base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dummy_data.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "llama_prefix_finetune")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "llama_prefix_final_model")

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# === Callback to log loss ===
class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            with open(os.path.join(LOG_DIR, "loss_log.csv"), "a") as f:
                f.write(f"{state.global_step},{logs['loss']:.4f}\n")

def train():
    if HUGGINGFACE_TOKEN is None:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HUGGINGFACE_TOKEN
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 2. Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # 3. Load base model with LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        token=HUGGINGFACE_TOKEN
    )

    base_model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for new pad token

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    llama_model = get_peft_model(base_model, lora_config)

    # 4. Add prefix adapter for numeric features
    adapter = FeaturePrefixAdapter(
        input_dim=FEATURE_DIM,
        hidden_dim=256,
        output_dim=EMBEDDING_DIM,
        num_tokens=PREFIX_TOKEN_COUNT
    )
    full_model = PrefixLLaMAModel(llama_model, adapter)

    # 5. Load dataset (updated)
    features, fatigue_levels, responses = input_process.load_csv_dataset(CSV_PATH)
    dataset = input_process.SensorTextDataset(features, fatigue_levels, responses, tokenizer)

    # 6. Training arguments with TensorBoard support
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_dir=LOG_DIR,
        logging_steps=10,
        report_to="tensorboard",
        save_total_limit=1,
        bf16=True  # Change to False if unsupported
    )

    # 7. Trainer setup
    trainer = Trainer(
        model=full_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=input_process.custom_collate,
        callbacks=[LossLoggerCallback()]
    )

    # 8. Train
    trainer.train()

    # 9. Save model artifacts
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    torch.save(adapter.state_dict(), os.path.join(FINAL_MODEL_DIR, "prefix_adapter.pth"))

    with open(os.path.join(FINAL_MODEL_DIR, "config.txt"), "w") as f:
        f.write(f"Model: {MODEL_NAME}\nEpochs: 3\nBatch size: {BATCH_SIZE}\nPrefix shape: ({PREFIX_TOKEN_COUNT}, {EMBEDDING_DIM})")

if __name__ == "__main__":
    train()
