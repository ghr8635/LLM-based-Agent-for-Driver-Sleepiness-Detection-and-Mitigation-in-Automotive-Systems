#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from ament_index_python.packages import get_package_share_directory
import os


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from datetime import datetime

from llm_fatigue_handling.model_wrapper_with_mlp_adapter import FeaturePrefixAdapter, PrefixLLaMAModel
from llm_fatigue_handling.input_process import build_driver_state_prompt_from_list

PACKAGE_NAME = 'llm_fatigue_handling'
PACKAGE_DIR = get_package_share_directory(PACKAGE_NAME)
MODEL_PATH = "./llama_prefix_final_model"
MODEL_PATH = os.path.join(PACKAGE_DIR, 'llama_prefix_final_model')
FEATURE_DIM = 12

class LLMNode(Node):

    def __init__(self):
        super().__init__('llm_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'feature_vector',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'llm_response', 10)

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', load_in_4bit=True)

        # Load prefix adapter
        adapter = FeaturePrefixAdapter()
        adapter.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'prefix_adapter.pth')))
        self.model = PrefixLLaMAModel(base_model, adapter)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.get_logger().info("LLaMA Inference Node is ready.")

    def listener_callback(self, msg):
        features = list(msg.data)

        if len(features) != 12:
            self.get_logger().error(f"Expected 12 features, got {len(features)}")
            return
        
        prompt = build_driver_state_prompt_from_list(features)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.llama.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_msg = String()
        response_msg.data = response
        self.publisher.publish(response_msg)
        self.get_logger().info("Published LLM response.")

        # Save to JSON
        self.save_to_json(response)

    def save_to_json(self, response):
        record = {
        "timestamp": datetime.now().isoformat(),
        "response": response
        }
        output_dir = os.path.join(PACKAGE_DIR, 'llm_outputs')
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.join(output_dir, f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_name, 'w') as f:
            json.dump(record, f, indent=2)
        self.get_logger().info(f"Saved output to {file_name}")


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()