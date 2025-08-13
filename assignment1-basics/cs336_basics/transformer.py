import torch
from torch import nn, FloatTensor
from einops import rearrange, einsum, repeat

from jaxtyping import Float, Int
from torch import Tensor, LongTensor

from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwishGLU import SwishGLU


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()

        self.in_ = in_features
        self.out_ = out_features
        sigma = torch.math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            ),
        )


    def forward(self, x: Float[Tensor, "... d_in"]) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        sigma = torch.math.sqrt(2 / (num_embeddings + embedding_dim))
        self.embeddings = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), mean=0, std=sigma, a=-3 * sigma, b=3 * sigma),
        )

    def forward(self, x: Float[LongTensor, "... sequence_length"]) -> torch.Tensor:

        return self.embeddings[x]


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device =None):
        super(RoPE, self).__init__()
        """
        Since we only care about the relative rotation of tokens within a given sequence, 
        we can reuse the values we compute for cos(θi,k ) and sin(θi,k ) across layers, and different batches.
        """
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.R = self.generate_rotation_block(max_seq_len, d_k)


    ## ref: https://github.com/Spectual/stanford-cs336-a1/blob/main/cs336_basics/RotaryPositionalEmbedding.py
    def generate_rotation(self, k, i, d_k):
        theta_i_k = i / (self.theta ** (2 * k / d_k))
        cos = torch.math.cos(theta_i_k)
        sin = torch.math.sin(theta_i_k)
        return torch.Tensor([
            [cos, -sin],
            [sin, cos]
        ])

    def generate_rotation_block(self, max_seq, d_k):

        res = torch.zeros(max_seq, d_k, d_k)
        for i in range(max_seq):
            blocks = [self.generate_rotation(k, i, d_k) for k in range(d_k // 2 )]
            res[i, :, :] = torch.block_diag(*blocks)
        return res



    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_position: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        print(f"RoPE: x.shape:{x.shape}, token_position.shape: {token_position.shape}, R:{self.R[token_position].shape}")

        return einsum(x, self.R[token_position], "... d_k, ... d_e d_k -> ... d_e")


class MultiHeadSelfAttention(nn.Module):

    # here is the feature we should implement
    # 1. apply RoPE to Q, K.
    # 2. apply casual mask
    # 3. treat heads as the batch dimension

    def __init__(self, num_heads: int, d_model, use_rope:bool, d_head=0, max_seq_len=0, theta=0, device=None, dtype=None, token_position=None):
        super(MultiHeadSelfAttention, self).__init__()
        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_k

        sigma = torch.math.sqrt(2 / (d_model + d_k))
        self.Q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_k * num_heads),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        )
        self.K = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_k * num_heads),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        )
        self.V = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_v * num_heads),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        )

        self.O = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_v * num_heads),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        )

        self.RoPE = RoPE(theta, d_head, max_seq_len, device) if use_rope else None
        self.token_positions = token_position
        self.mask = torch.where(torch.triu(
            torch.ones(num_heads, d_v , d_v ), diagonal=0
        ).transpose(-2, -1) == 1, True, False)

    def forward(self, x: Float[Tensor, "... seq_len d_in"]) -> Float[Tensor, "... seq_len d_out"]:
        # d_in = d_model = d_out
        print(f"mha: x:{x.shape}")
        q = rearrange(einsum(x, self.Q, "... d_in, hd_k d_in  -> ... hd_k"), "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        k = rearrange(einsum(x, self.K, "... d_in, hd_k d_in  -> ... hd_k"), "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        v = rearrange(einsum(x, self.V, "... d_in, hd_v d_in  -> ... hd_v"), "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)

        if self.RoPE is not None:
            q = self.RoPE(q, self.token_positions)
            k = self.RoPE(k, self.token_positions)

        mask = torch.where(torch.triu(
            torch.ones(q.shape[-2] , k.shape[-2] ), diagonal=0
        ).T == 1, True, False)[None,None,:,:]

        o = rearrange(scaled_dot_product_attention(q, k, v, mask), "... heads seq_len d_v -> ... seq_len (heads d_v)")  # Float[Tensor, "... heads seq_len d_v"
        print(f"mha: o:{o.shape}, O weight: {self.O.shape}")
        return einsum(o, self.O, "... seq_len d_out, d_model d_out-> ... seq_len d_model")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, use_rope=False, device=None, dtype=None):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadSelfAttention(num_heads, d_model,
                                          use_rope=use_rope, d_head = d_model // num_heads, max_seq_len=max_seq_len,
                                          theta=theta, device=device, dtype=dtype)
        self.swiGlu = SwishGLU(d_ff, d_model, device=device, dtype=dtype)
        self.RMSNorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.RMSNorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_in"]):
        seq_len = x.shape[-2]
        self.mha.token_positions = torch.arange(1, seq_len + 1)[None,:]
        layer1 = x + self.mha(self.RMSNorm1(x))
        # layer1 + self.swiGlu(self.RMSNorm(layer1))
        return layer1 + self.swiGlu(self.RMSNorm2(layer1))

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, vocab_size, context_length, d_ff, theta,use_rope=False, device=None, dtype=None, num_layers=0):
        super(Transformer, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta, use_rope, device, dtype) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size)
        self.softmax = SoftMax()

    def forward(self, x: Float[Tensor, "... seq_len d_in"]):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x



def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = ( einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")  ) / torch.math.sqrt(d_k)
    print(f"sdpa: Q:{Q.shape}, K:{K.shape}, V:{V.shape},pre_softmax:{pre_softmax.shape}, mask{mask.shape}")
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
    qk_ = softmax(pre_softmax)
    return einsum(qk_, V, "... queries keys, ... keys d_v -> ... queries d_v")



class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x, dim=-1):
        return softmax(x, dim)
        # nor_ = (x - x.max(dim, keepdim=True).values).exp()
        # return nor_ / nor_.sum(dim=dim, keepdim=True)

def softmax(x: torch.Tensor, dim=-1):
    nor_ = (x - x.max(dim, keepdim=True).values).exp()
    return nor_ / nor_.sum( dim=dim, keepdim=True)