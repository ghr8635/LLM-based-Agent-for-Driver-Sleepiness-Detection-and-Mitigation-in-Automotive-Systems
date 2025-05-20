import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List
import numpy as np


# Structured Prompt Function
def build_driver_state_prompt_from_list(features: list) -> str:
    if len(features) != 12:
        raise ValueError(f"Expected 12 input features, got {len(features)}")

    (
        perclos, blink_rate, yawning_rate, head_nodding_rate,
        steering_entropy, srr, sav, sdlp, lane_departure_freq,
        fatigue_cam, fatigue_steering, fatigue_lane
    ) = features

    prompt = f"""
You are an intelligent in-cabin assistant. Based on the following driving behavior and fatigue indicators, generate an appropriate intervention to help the driver stay alert.

<vision_features>
perclos: {perclos:.2f}%  
blink_rate: {blink_rate:.1f} per minute  
yawning_rate: {yawning_rate:.1f} per minute  
head_nodding_rate: {head_nodding_rate:.1f} per minute  
</vision_features>

<steering_features>
steering_entropy: {steering_entropy:.3f}  
steering_reversal_rate: {srr:.1f} per minute  
steering_angle_variability: {sav:.2f}°  
</steering_features>

<lane_features>
sdlp: {sdlp:.2f} m  
lane_departure_frequency: {lane_departure_freq:.1f} per minute  
</lane_features>

<fatigue_scores>
fatigue_camera: {fatigue_cam:.1f}%  
fatigue_steering: {fatigue_steering:.1f}%  
fatigue_lane: {fatigue_lane:.1f}%  
</fatigue_scores>

<Expected Intervention>
Based on the above signals, what should be the appropriate intervention?
""".strip()
    return prompt


MAX_LENGTH = 256
# Dataset Class
class SensorTextDataset(Dataset):
    def __init__(self, features: List[List[float]], responses: List[str], tokenizer):
        self.features = features
        self.responses = responses
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        prompt = build_driver_state_prompt_from_list(feature_vector)
        response = self.responses[idx]

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
        labels = self.tokenizer(response, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "features": torch.tensor(feature_vector, dtype=torch.float32)
        }

# Collate Function
def custom_collate(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch])
    }
