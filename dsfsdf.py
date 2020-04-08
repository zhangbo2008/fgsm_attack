from torch import nn
import torch
import math

q1 = torch.rand(4,  42, 64)
q2 = torch.rand(4,  64, 128)
ret = q1@q2
print(ret.shape)






























