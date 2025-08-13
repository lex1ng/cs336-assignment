import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor, LongTensor


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        sigma = torch.math.sqrt(2 / d_model)
        self.g = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_model),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma,
            )
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> torch.Tensor:
        d_type = x.dtype
        x.to(torch.float32)
        rms = torch.sqrt( (1 / self.d_model ) * torch.pow(x, 2).sum(dim=-1, keepdim=True)  + self.eps)
        res = x * self.g / rms
        return res.to(d_type)
