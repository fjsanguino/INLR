import torch.nn as nn
import torch

from utils import INRF_B

def INRF_IQ(output, target):

    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(INRF_B(output), INRF_B(target)))

    return loss
