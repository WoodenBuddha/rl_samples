import torch
import torch.nn as nn
import numpy as np

class PolicyGradientAgent(nn.Module):
    model = None

    def __init__(self):
        super(PolicyGradientAgent, self).__init__()
        self.dense = nn.Linear()

    def build(self):
        model = None

    def forward(self, x):
        x = None

    def policy_function(self, x):
        pass
