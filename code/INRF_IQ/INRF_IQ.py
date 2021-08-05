import torch.nn as nn
import torch

from utils import INRF_B

class INRF_IQ(nn.Module):

    def __init__(self):
        super(INRF_IQ, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, output, target):
        loss = torch.sqrt(self.mse(INRF_B(output, mode='tensor'), INRF_B(target, mode='tensor')))
        return loss


