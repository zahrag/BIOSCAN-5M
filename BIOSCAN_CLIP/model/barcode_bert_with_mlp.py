import torch
import torch.nn as nn


class BarcodeBERT_And_MLP(nn.Module):
    def __init__(self, barcode_bert, mlp):
        super(BarcodeBERT_And_MLP, self).__init__()
        self.barcode_bert = barcode_bert
        self.mlp = mlp
        for param in self.barcode_bert.parameters():
            param.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            feature = self.barcode_bert(x).hidden_states[-1].mean(dim=1).squeeze()
        return self.mlp(feature)