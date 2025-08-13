

import os
from typing import Callable, Any

import torch
import triton
import triton.language as tl
from einops import einsum

from einops import einsum, rearrange, reduce


class FlashAttentionPytorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: [Nq, d], K,V: [Nk, d], tile sizs Bq, Bk
        # print(f"input, Q:{Q.shape}, K:{K.shape}, V:{V.shape}, is_causal:{is_causal}")
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape
        # Nq, Nk = Q.shape[-2], K.shape[-2]
        Bq, Bk = 16, 16
        Tq, Tk = (Nq + Bq -1) // Bq, ( Nk + Bk - 1 ) // Bk
        # print(f"Bq: {Bq}, Bk: {Bk}, Tq:{Tq}, Tk:{Tk}")
        Q_tiles = torch.split(Q, Bq, -2)

        K_tiles = torch.split(K, Bk, -2)
        V_tiles = torch.split(V, Bk, -2)

        O = []
        L = []

        for i in range(Tq):
            Oi_pre = torch.zeros((B, Bq, d))
            li_pre = torch.zeros((B, Bq))
            mi_pre = torch.full((B, Bq,), float('-inf'))

            tile_Q = Q_tiles[i]
            # print(f"tiles, Q:{tile_Q.shape}")
            for j in range(Tk):
                tile_K = K_tiles[j]
                tile_V = V_tiles[j]
                # print(f"tiles, K:{tile_K.shape}, V:{tile_V.shape}")
                Sij = einsum(tile_Q, tile_K, "... Bq d, ... Bk d -> ... Bq Bk") / torch.math.sqrt(d)
                # print(f"Sij:{Sij.shape}")
                # print(f"mi_pre:{mi_pre.shape}")
                mij = torch.max(mi_pre, Sij.max(-1).values)
                # print(f"mij:{mij.shape}")
                pij_unnor = torch.exp(Sij - mij.unsqueeze(-1))
                # print(f"pij_unnor:{pij_unnor.shape}")
                # print(f"li_pre:{li_pre.shape}")
                # print(f"pij_unnor_sum: {pij_unnor.sum(-1).shape}")
                # print(f"test: {(torch.exp(mi_pre - mij) * li_pre ).shape}")
                lij = torch.exp(mi_pre - mij) * li_pre + pij_unnor.sum(-1)
                # print(f"lij: {lij.shape}")
                pv = einsum(pij_unnor, tile_V, "... Bq Bk, ... Bk d -> ... Bq d")
                # print(f"test1: {torch.diag_embed(torch.exp(mi_pre - mij)).sum(-1).shape}, pv: {pv.shape}")
                # Oij = torch.diag_embed(torch.exp(mi_pre - mij)).sum(-1, keepdim=True) * Oi_pre + pv
                Oij = torch.exp(mi_pre - mij).unsqueeze(-1) * Oi_pre + pv

                li_pre = lij
                mi_pre = mij
                Oi_pre = Oij

            Oi = torch.inverse(torch.diag_embed(li_pre)).sum(-1, keepdim=True) * Oi_pre
            Li = mi_pre + torch.log(li_pre)

            O.append(Oi)
            L.append(Li)

        O = torch.cat(O, 1)
        L = torch.cat(L, 1)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx: Any, dO) -> Any:
        L, Q, K, V, O = ctx.saved_tensors
        D = torch.sum(O * dO, -1)
        d = Q.shape[-1]
        S = einsum(Q, K, "... sq d, ... sk d -> ... sq sk") / torch.math.sqrt(d)

        P = torch.exp(S - L.unsqueeze(-1))

        dV = einsum(P, dO, "... sq sk, ... sq d -> ... sk d")
        dP = einsum(dO, V, "... sq d, ... sk d -> ... sq sk")
        dS = P * (dP - D.unsqueeze(-1))
        dQ = einsum(dS, K, "... sq sk, ... sk d -> ... sq d") / torch.math.sqrt(d)
        dK = einsum(dS, Q, "... sq sk, ... sq d -> ... sk d") / torch.math.sqrt(d)

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,         # Nq, Nk
    scale,                     # 1 / sqrt(D)
    D: tl.constexpr,           # D
    Q_TILE_SIZE: tl.constexpr, # = Bq
    K_TILE_SIZE: tl.constexpr, # = Bk
    is_casual:  tl.constexpr,
):
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)
  # print("pid:", query_tile_index)
  # print("batch:", batch_index)
  Q_block_ptr = tl.make_block_ptr(
      Q_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      K_ptr + batch_index * stride_kb,
      shape=(N_KEYS, D),
      strides=(stride_kk, stride_kd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  V_block_ptr = tl.make_block_ptr(
      V_ptr + batch_index * stride_vb,
      shape=(N_KEYS, D),
      strides=(stride_vk, stride_vd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1,0)
  )

  O_block_ptr = tl.make_block_ptr(
      O_ptr + batch_index * stride_ob,
      shape=(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(query_tile_index *Q_TILE_SIZE, 0 ),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  L_block_ptr = tl.make_block_ptr(
      L_ptr + batch_index * stride_lb,
      shape=(N_QUERIES,),
      strides=(stride_lq,),
      offsets=(query_tile_index * Q_TILE_SIZE, ),
      block_shape=(Q_TILE_SIZE,),
      order=(0, ),

  )


  tile_Q = tl.load(Q_block_ptr,boundary_check=(0, 1), padding_option="zero").to(tl.float32)


  Oi_pre = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
  li_pre = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
  mi_pre = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

  q_global_start_idx = query_tile_index * Q_TILE_SIZE
  q_local_idx = tl.arange(0, Q_TILE_SIZE)
  # print("\ntile_Q:", tile_Q.shape)
  for j in range( tl.cdiv(N_KEYS, K_TILE_SIZE)):

    tile_K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # print("tile_K:", tile_K.shape)
    # print("tile_Q type", tile_Q.type.is_block())
    # print("tile_K type", tile_K.T.type.is_block())
    k_global_start_idx = j * K_TILE_SIZE
    k_local_idx = tl.arange(0, K_TILE_SIZE)
    Sij = tl.dot(tile_Q, tl.permute(tile_K, (1, 0))) * scale
    if is_casual:
        Sij = tl.where(
            (q_global_start_idx + q_local_idx)[:,None] >= (k_global_start_idx + k_local_idx )[None,:],
            Sij,
            -1e6
        )
    # print("Sij:", Sij.shape)
    # print("mi_pre shape:",mi_pre.shape ,"Sij max :", Sij.max(-1).shape)
    # print(f"mi_pre: {mi_pre}\nSij max: {Sij.max(-1)}\n" )
    mij = tl.maximum(mi_pre, Sij.max(-1))
    # print("mij:", mij.shape)
    pij_unnor = tl.exp(Sij - tl.expand_dims(mij,-1)).to(tile_V.dtype)
    # print("pij_unnor:", pij_unnor.shape)
    lij = tl.exp(mi_pre - mij) * li_pre + pij_unnor.sum(-1)
    # print("lij:", lij.shape)
    pv = tl.dot(pij_unnor, tile_V)
    # print("pv:", pv.shape)
    # print("tmp shape:", tl.exp(mi_pre - mij)[: None] .shape)
    # print("tmp:", tl.exp(mi_pre - mij)[: None] )
    # print("Oi_pre shape:", Oi_pre.shape)
    # print("Oi_pre:", Oi_pre)
    # print(f"tl.shpae:{tl.expand_dims(mi_pre - mij, axis=1).shape}")
    # print(f"tl.exp.shape:{tl.exp(tl.expand_dims(mi_pre - mij, axis=1)).shape}")
    # Oij = tl.exp(tl.expand_dims(mi_pre - mij, axis=1)) * Oi_pre + pv
    Oij = tl.expand_dims(tl.exp(mi_pre - mij), axis=1) * Oi_pre + pv
    # print("Oij:", Oij.shape)
    li_pre = lij
    mi_pre = mij
    Oi_pre = Oij

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  # print(f"out loop li_pre:{li_pre.shape}, Oi_pre: {Oi_pre.shape}")
  # print("before shape:", (tl.expand_dims((1.0 / li_pre), axis=1) * Oi_pre ).shape )
  # print("before :", tl.expand_dims((1.0 / li_pre), axis=1) * Oi_pre )
  Oi = (tl.expand_dims((1.0 / li_pre), axis=1) * Oi_pre )
  # print("Oi shape,:", Oi.shape)
  # print("Oi,:", Oi)
  Li =  mi_pre + tl.log(li_pre)
  # print("Li shape,:", Li.shape)
  # print("Li,:", Li)


  # tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
  tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
  # tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0))
  tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod, # new
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,         # Nq, Nk
    scale,                     # 1 / sqrt(D)
    D: tl.constexpr,           # D
    Q_TILE_SIZE: tl.constexpr, # = Bq
    K_TILE_SIZE: tl.constexpr, # = Bk
    is_casual:  tl.constexpr,
):
  key_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)
  # print("pid:", query_tile_index)
  # print("batch:", batch_index)
  Q_block_ptr = tl.make_block_ptr(
      Q_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(0, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  dQ_block_ptr = tl.make_block_ptr(
      dQ_ptr + batch_index * stride_qb,
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(0, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      K_ptr + batch_index * stride_kb,
      shape=(N_KEYS, D),
      strides=(stride_kk, stride_kd),
      offsets=(key_tile_index * K_TILE_SIZE , 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  dK_block_ptr = tl.make_block_ptr(
      dK_ptr + batch_index * stride_kb,
      shape=(N_KEYS, D),
      strides=(stride_kk, stride_kd),
      offsets=(key_tile_index * K_TILE_SIZE , 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  V_block_ptr = tl.make_block_ptr(
      V_ptr + batch_index * stride_vb,
      shape=(N_KEYS, D),
      strides=(stride_vk, stride_vd),
      offsets=(key_tile_index * K_TILE_SIZE , 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1,0)
  )
  dV_block_ptr = tl.make_block_ptr(
      dV_ptr + batch_index * stride_vb,
      shape=(N_KEYS, D),
      strides=(stride_vk, stride_vd),
      offsets=(key_tile_index * K_TILE_SIZE , 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1,0)
  )
  O_block_ptr = tl.make_block_ptr(
      O_ptr + batch_index * stride_ob,
      shape=(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(0 , 0 ),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  dO_block_ptr = tl.make_block_ptr(
      dO_ptr + batch_index * stride_dob,
      shape=(N_QUERIES, D),
      strides=(stride_doq, stride_dod),
      offsets=(0 , 0 ),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
  )
  L_block_ptr = tl.make_block_ptr(
      L_ptr + batch_index * stride_lb,
      shape=(N_QUERIES,),
      strides=(stride_lq,),
      offsets=(0, ),
      block_shape=(Q_TILE_SIZE,),
      order=(0, ),
  )

  D_block_ptr = tl.make_block_ptr(
      D_ptr + batch_index * stride_lb,
      shape=(N_QUERIES,),
      strides=(stride_lq,),
      offsets=(0, ),
      block_shape=(Q_TILE_SIZE,),
      order=(0, ),
  )



  tile_K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
  tile_V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

  dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
  dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

  k_global_start_idx = key_tile_index * K_TILE_SIZE
  k_local_idx = tl.arange(0, K_TILE_SIZE)
  # print("\ntile_Q:", tile_Q.shape)
  for i in range( tl.cdiv(N_QUERIES, Q_TILE_SIZE)):

    tile_Q  = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_dQ = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_O  = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tile_D = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    # print("tile_K:", tile_K.shape)
    # print("tile_Q type", tile_Q.type.is_block())
    # print("tile_K type", tile_K.T.type.is_block())

    Sij = tl.dot(tile_Q, tl.permute(tile_K, (1, 0))) * scale
    print(f"Sij shape:{Sij.shape}, tile_L.shape:{tile_L.shape}",)
    Pij = tl.exp(Sij - tl.expand_dims(tile_L, 1))

    dV += tl.dot(tl.permute(Pij, (1, 0)), tile_dO)
    dPij = tl.dot(tile_dO, tl.permute(tile_V, (1, 0)))
    dSij = ( Pij * ( dPij - tl.expand_dims(tile_D, 1) ) ) * scale

    tile_dQ += tl.dot(dSij, tile_K)
    tl.store(dQ_block_ptr, tile_dQ, boundary_check=(0, 1))

    dK += tl.dot(tl.permute(dSij, (1, 0)), tile_Q)

    Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
    dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
    O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
    dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE))
    D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE))


  tl.store(dK_block_ptr, dK, boundary_check=(0, 1))
  tl.store(dV_block_ptr, dV, boundary_check=(0, 1))



class FlashAttentionTriton(torch.autograd.Function):

  @staticmethod
  def forward(ctx, Q, K, V, is_casual=False):
    B, Nq, d = Q.shape
    _, Nk, _ = K.shape
    ctx.is_casual = is_casual

    Bq, Bk = 16, 16
    Tq, Tk = (Nq + Bq - 1) // Bq,  (Nk + Bk - 1) // Bk

    O = torch.empty((B, Nq, d), device="cuda:0")
    L = torch.empty((B, Nq), device="cuda:0")
    # print(f"scale: {1.0/ torch.math.sqrt(d)}")
    flash_fwd_kernel[(Tq, B)](
        Q_ptr=Q, K_ptr=K, V_ptr=V,
        O_ptr=O, L_ptr=L,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
        stride_lb=L.stride(0), stride_lq=L.stride(1),
        N_QUERIES=Nq, N_KEYS=Nk,         # Nq, Nk
        scale= 1.0 / torch.math.sqrt(d),                     # 1 / sqrt(D)
        D=d,           # D
        Q_TILE_SIZE=Bq, # = Bq
        K_TILE_SIZE=Bk, # = Bk
        is_casual=is_casual,
    )
    ctx.is_casual =  is_casual
    ctx.save_for_backward(L, Q, K, V, O)

    return O

  @staticmethod
  def backward(ctx: Any, dO) -> Any:

      L, Q, K, V, O = ctx.saved_tensors
      D = torch.sum(  dO * O, -1)
      # d = Q.shape[-1]
      B, Nq, d = Q.shape
      _, Nk, _ = K.shape
      Bq, Bk = 16, 16
      Tq, Tk = (Nq + Bq - 1) // Bq, (Nk + Bk - 1) // Bk

      dQ = torch.empty_like(Q)
      dK = torch.empty_like(K)
      dV = torch.empty_like(V)

      flash_bwd_kernel[(Tk, B)](
          Q_ptr=Q, K_ptr=K, V_ptr=V,
          L_ptr=L, O_ptr=O, dO_ptr=dO, dQ_ptr=dQ, dK_ptr=dK, dV_ptr=dV,D_ptr=D,
          stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
          stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
          stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
          stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
          stride_dob=dO.stride(0), stride_doq=dO.stride(1), stride_dod=dO.stride(2),
          stride_lb=L.stride(0), stride_lq=L.stride(1),
          N_QUERIES=Nq, N_KEYS=Nk,  # Nq, Nk
          scale=1.0 / torch.math.sqrt(d),  # 1 / sqrt(D)
          D=d,  # D
          Q_TILE_SIZE=Bq,  # = Bq
          K_TILE_SIZE=Bk,  # = Bk
          is_casual=ctx.is_casual,
      )
      return dQ, dK, dV, None








