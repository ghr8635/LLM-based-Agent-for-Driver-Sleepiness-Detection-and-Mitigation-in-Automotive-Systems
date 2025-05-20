import torch
import torch.nn as nn

FEATURE_DIM = 12
EMBEDDING_DIM = 4096
PREFIX_TOKEN_COUNT = 5

# MLP Adapter
class FeaturePrefixAdapter(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=256, output_dim=EMBEDDING_DIM, num_tokens=PREFIX_TOKEN_COUNT):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tokens * output_dim)
        )

    def forward(self, x):
        prefix = self.mlp(x)
        return prefix.view(-1, self.num_tokens, self.output_dim)

# Wrapper Model
class PrefixLLaMAModel(nn.Module):
    def __init__(self, llama_model, adapter):
        super().__init__()
        self.llama = llama_model
        self.adapter = adapter
        self.embedding_layer = llama_model.model.embed_tokens

    def forward(self, input_ids, attention_mask, features, labels=None):
        text_embed = self.embedding_layer(input_ids)
        prefix_embed = self.adapter(features).to(text_embed.dtype)
        full_embed = torch.cat([prefix_embed, text_embed], dim=1)
        prefix_mask = torch.ones(features.size(0), prefix_embed.size(1)).to(attention_mask.device)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.llama(inputs_embeds=full_embed, attention_mask=full_mask, labels=labels)