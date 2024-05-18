import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import math



class MLPVersionCLIP(nn.Module):
    def __init__(self, image_input_dim=512, dna_input_dim=768, hidden_dim=512, output_dim=512):
        super(MLPVersionCLIP, self).__init__()
        self.image_feature_encoder = MLPEncoder(input_dim=image_input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.dna_feature_encoder = MLPEncoder(input_dim=dna_input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, image_feature, dna_feature):
        return F.normalize(self.image_feature_encoder(image_feature), dim=-1), F.normalize(
            self.dna_feature_encoder(dna_feature), dim=-1
        )


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)
