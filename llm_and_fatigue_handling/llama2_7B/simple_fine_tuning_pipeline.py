import os
from huggingface_hub import login

# === Step 1: Authenticate and configure PEFT compatibility ===
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
login(token=HUGGINGFACE_TOKEN)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import simple_input_process
from simple_faiss_vd import PrefixVectorDB
import random
import numpy as np
import shutil

# === Configuration ===
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
FEATURE_DIM = 9
MAX_LENGTH = 192
BATCH_SIZE = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "captured_data.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "llama_prefix_finetune")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "llama_prefix_final_model")

# === Reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === Callback ===
class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            with open(os.path.join(LOG_DIR, "loss_log.csv"), "a") as f:
                f.write(f"{state.global_step},{logs['loss']:.4f}\n")

def train():
    # === 1. Load Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HUGGINGFACE_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # === 2. Quantization Config ===
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # === 3. Load Base + LoRA Model ===
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        token=HUGGINGFACE_TOKEN
    )
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    full_model = get_peft_model(base_model, lora_config)

    # === 4. Load Dataset ===
    features, fatigue_levels, responses = simple_input_process.load_csv_dataset(CSV_PATH)
    dataset = simple_input_process.SensorTextDataset(features, fatigue_levels, responses, tokenizer, prefix_token_count=0)

    # === 5. Save Feature Vectors to FAISS DB ===
    print("Extracting and saving 9D feature vectors to FAISS database...")
    db = PrefixVectorDB(dim=FEATURE_DIM)
    for i in range(len(dataset)):
        item = dataset[i]
        vector = item["features"].numpy()
        intervention = responses[i]
        db.add_vector(vector, intervention, source='fine_tuning')
    db.save()
    print("✅ Vector database populated with 9D vectors.")

    # === 6. Training Args ===
    training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      per_device_train_batch_size=BATCH_SIZE,
      num_train_epochs=0.01,
      learning_rate=5e-5,
      save_strategy="no",
      logging_dir=LOG_DIR,
      logging_steps=10,
      report_to="tensorboard",
      bf16=True,
      remove_unused_columns=False  # <-- 🔥 CRITICAL FIX
  )

    # === 7. Trainer ===
    trainer = Trainer(
        model=full_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=simple_input_process.custom_collate,
        callbacks=[LossLoggerCallback()]
    )

    # === 8. Train ===
    trainer.train()

    # === 9. Save Artifacts ===
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    full_model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    with open(os.path.join(FINAL_MODEL_DIR, "config.txt"), "w") as f:
        f.write(f"Model: {MODEL_NAME}\nEpochs: 1\nBatch size: {BATCH_SIZE}\nVector dimension: {FEATURE_DIM}")

if __name__ == "__main__":
    train()
