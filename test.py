#from datasets import load_dataset

#ds = load_dataset("Exorde/exorde-social-media-december-2024-week1")

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))