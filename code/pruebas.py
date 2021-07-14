import numpy as np
import torch
import torch.nn as nn


m = SynthesisTransformer().float().cuda()

torch_tensor = torch.from_numpy(np.random.rand(16,128,96,96)).cuda()
out = m(torch_tensor.float())
print(out.shape)