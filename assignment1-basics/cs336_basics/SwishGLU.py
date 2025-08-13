

from torch import nn
import torch
from jaxtyping import Float, Int
from torch import Tensor, LongTensor
from einops import rearrange, einsum


class SiLu(nn.Module):
    def __init__(self, ):
        super(SiLu, self).__init__()

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_in"]:
        return x * torch.sigmoid(x)

class SwishGLU(nn.Module):

    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super(SwishGLU, self).__init__()

        self.d_ff = d_ff
        self.d_model = d_model

        sigma = torch.math.sqrt(2 / (self.d_ff + self.d_model))
        self.silu = SiLu()
        self.W1 = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_ff, d_model),
                mean=0, std=sigma, a=-3*sigma, b = 3 * sigma),
        )
        self.W3 = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_ff, d_model),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma),
        )
        self.W2 = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty( d_model, d_ff),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma),
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_model"]:

        w1x = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        swish = self.silu(w1x)
        in_ = swish * w3x

        return einsum(in_, self.W2, "... d_ff, d_model d_ff -> ... d_model")





