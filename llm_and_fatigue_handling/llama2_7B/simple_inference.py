import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from peft import PeftModel
from simple_faiss_vd import runtime_add, retrieve_similar_vectors

# === Authenticate Hugging Face ===
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
login(token=HUGGINGFACE_TOKEN)

# === Config ===
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_DIR = "./llama_prefix_final_model"
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HUGGINGFACE_TOKEN)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"

# === Quantization Config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# === Load Base Model ===
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    token=HUGGINGFACE_TOKEN,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

# === Fix: Resize embeddings to match tokenizer size
base_model.resize_token_embeddings(len(tokenizer))

# === Load LoRA adapter ===
llama_model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    device_map="auto",
    use_auth_token=False
)
llama_model.eval()

# === Input Features and Fatigue Levels ===
features = [24, 8, 0.38, 0.23, 96.0, 0.4, 0.21, 8.0, 1.3]  # example numeric features
fatigue_levels = ["low", "medium", "medium"]

# === FAISS Retrieval (Optional) ===
feature_vector = np.array(features, dtype=np.float32)
results = retrieve_similar_vectors(feature_vector, k=3)
retrieved_interventions = [
    meta.get("intervention") for _, meta, _ in results
    if meta.get("intervention") and meta.get("intervention").strip().lower() not in {"", "none", "driver alert"}
]

context = ""
if retrieved_interventions:
    context = "Previously suggested interventions for similar scenarios: " + "; ".join(retrieved_interventions) + ". "

# === Build Prompt ===
prompt = f"""
You are an intelligent in-cabin assistant.

Fatigue levels:
- Camera: {fatigue_levels[0]}
- Steering: {fatigue_levels[1]}
- Lane: {fatigue_levels[2]}

Driving behavior features:
- Blink rate: {features[0]:.1f} per minute
- Yawning rate: {features[1]:.1f} per minute
- PERCLOS: {features[2]:.2f}%
- SDLP: {features[3]:.2f} m
- Lane keeping ratio: {features[4]:.1f}
- Lane departure frequency: {features[5]:.1f} per minute
- Steering entropy: {features[6]:.2f}
- Steering reversal rate: {features[7]:.1f} per minute
- Steering angle variability: {features[8]:.2f}°

{context}Based on the above driver state and past examples, suggest **only one** intervention to keep the driver alert.

⚠️ IMPORTANT: You must output in this fixed format — no extra text, no repetition.

Fan: Level X      ← X is a number like 1, 2, or 3  
Music: On/Off  
Vibration: On/Off  
Reason: <short explanation of the logic>

Example output:
Fan: Level 2  
Music: On  
Vibration: Off  
Reason: High blink rate and PERCLOS indicate moderate drowsiness.

Now, provide your single final intervention and stop.
""".strip()

# === Tokenize ===
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)

# === Generate ===
with torch.no_grad():
    output = llama_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id  # Stops at EOS
    )

# === Decode
response = tokenizer.decode(output[0], skip_special_tokens=True)

# === Postprocess: Extract only the first 'Fan:' block to remove repetition
if response.count("Fan:") > 1:
    first_block = response.split("Fan:")[1]
    if "Reason:" in first_block:
        reason_part = first_block.split("Reason:")
        first_block = reason_part[0] + "Reason:" + reason_part[1].split("\n")[0]
    final_response = "Fan:" + first_block.strip()
else:
    intervention_start = response.find("Fan:")
    final_response = response[intervention_start:].strip() if intervention_start != -1 else response.strip()

# === Output
print("\n=== Generated Intervention ===")
print(final_response)

# === Save to vector DB
runtime_add(feature_vector, intervention=final_response)
