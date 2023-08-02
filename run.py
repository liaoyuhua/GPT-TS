import torch

from src.model import GPT2TS

model_type = "gpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"
