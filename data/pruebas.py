from models import ResidualBlock
import numpy as np

import torch

m = ResidualBlock().float()

torch_tensor = torch.from_numpy(np.random.rand(16,128,12,12))
out = m(torch_tensor.float())