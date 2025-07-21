import torch
import pandas as pd
from typing import List
import os
import csv

# === Intervention Function from CSV Row ===
def get_intervention_from_row(row) -> str:
    # Log intervention
    file_exists = os.path.isfile("interventions_log.csv")
    with open("interventions_log.csv", mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["fan", "music", "vibration", "reason"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "fan": row["fan"],
            "music": row["music"],
            "vibration": row["vibration"],
            "reason": row["reason"]
        })

    return (
        f"Fan: {row['fan']}\n"
        f"Music: {row['music']}\n"
        f"Vibration: {row['vibration']}\n"
        f"Reason: {row['reason']}"
    )

# === Prompt Builder ===
def build_driver_state_prompt_from_list(features: list, fatigue_list: list) -> str:
    if len(features) != 9:
        raise ValueError(f"Expected 9 input features, got {len(features)}")

    (
        blink_rate, yawning_rate, perclos,
        sdlp, lane_keeping_ratio, lane_departure_freq,
        steering_entropy, srr, sav
    ) = features

    prompt = f"""
You are an intelligent in-cabin assistant. Based on the following driving behavior and fatigue indicators, generate an appropriate intervention to help the driver stay alert.

Strictly follow this format:
Fan: Level X      ← X is a number like 1, 2, or 3  
Music: On/Off  
Vibration: On/Off  
Reason: <short explanation of why this intervention is needed>

Example:
Fan: Level 2  
Music: On  
Vibration: Off  
Reason: High PERCLOS and blinking suggest moderate fatigue.

<vision_features>
blink_rate: {blink_rate:.1f} per minute  
yawning_rate: {yawning_rate:.1f} per minute  
perclos: {perclos:.2f}%  
</vision_features>

<lane_features>
sdlp: {sdlp:.2f} m  
lane_keeping_ratio: {lane_keeping_ratio:.1f}  
lane_departure_frequency: {lane_departure_freq:.1f} per minute  
</lane_features>

<steering_features>
steering_entropy: {steering_entropy:.1f}  
steering_reversal_rate: {srr:.1f} per minute  
steering_angle_variability: {sav:.2f}°  
</steering_features>

<fatigue_levels>
fatigue_camera: {fatigue_list[0]}  
fatigue_steering: {fatigue_list[1]}  
fatigue_lane: {fatigue_list[2]}  
</fatigue_levels>

<Expected Intervention>
""".strip()
    return prompt

# === Custom Dataset ===
MAX_LENGTH = 256

class SensorTextDataset(torch.utils.data.Dataset):
    def __init__(self, features: List[List[float]], fatigue_levels: List[List[str]], responses: List[str], tokenizer, prefix_token_count: int):
        self.features = features
        self.fatigue_levels = fatigue_levels
        self.responses = responses
        self.tokenizer = tokenizer
        self.prefix_token_count = prefix_token_count

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        fatigue_list = self.fatigue_levels[idx]
        response = self.responses[idx]

        prompt = build_driver_state_prompt_from_list(feature_vector, fatigue_list)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
        labels = self.tokenizer(response, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "features": torch.tensor(feature_vector, dtype=torch.float32)
        }

# === Custom Collate Function ===
def custom_collate(batch):
    for i, item in enumerate(batch):
        if "features" not in item:
            print(f"[COLLATE DEBUG] ❌ Missing 'features' in item {i}: {item.keys()}")
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch])
    }

# === CSV Loader ===
def load_csv_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    # Features
    feature_columns = [
        'Blink Rate', 'Yawning Rate', 'PERCLOS', 'SDLP',
        'Lane Keeping Ratio', 'Lane Departure Frequency',
        'Steering Entropy', 'SRR', 'SAV'
    ]
    features = df[feature_columns].values.tolist()

    # Fatigue levels (provided in CSV)
    fatigue_levels = df[['fatigue_camera_level', 'fatigue_steering_level', 'fatigue_lane_level']].values.tolist()

    # Intervention text from each row
    responses = [
        get_intervention_from_row(row)
        for _, row in df.iterrows()
    ]

    return features, fatigue_levels, responses
