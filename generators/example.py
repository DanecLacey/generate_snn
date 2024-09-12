import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from io import BytesIO
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import sys

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
try:
    output_file = sys.argv[1]
except:
    print("No output file path given for sys.argv[1].")
    quit()

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NeuralNetwork().to(device)

module = model.linear_relu_stack[2]
prune.random_unstructured(module, name="weight", amount=0.5)
pruned_module = (module.weight).detach().numpy()
target = BytesIO()
mmwrite(target, coo_matrix(pruned_module), precision=3)

with open(output_file, "wb") as f:
    f.write(target.getbuffer())