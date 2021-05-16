import torch
import torch.nn as nn
import torch.nn.functional as F
class L2_CAF(nn.Module):
    def __init__(self,spatial_dim):
        super().__init__()
        self.filter = nn.Parameter(torch.ones((spatial_dim,spatial_dim),dtype=torch.float32),requires_grad=True)

    def forward(self, A):
        # filter_norm = torch.norm(self.filter,dim=[0,1])
        # l2_filter = self.filter/filter_norm
        # l2_filter = self.filter
        return A * F.normalize(self.filter,dim=[0,1])



