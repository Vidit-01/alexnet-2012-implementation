from torch import nn,Tensor
import numpy as np

X = Tensor(np.random.randn(10))
d = nn.Dropout()
print(X)
print(d(X))
print(d(X))