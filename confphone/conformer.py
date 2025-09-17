import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from conformer_conv import ConformerConvModule

class PositionalEncoding(nn.Module):
    """
    Computes positional encodings as:
      PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
      PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    """
    def __init__(self, d_model: int = 128, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]


class LayerNorm(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        ln_out = (x - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, atten_mask=None):
        # q: [B, n_head, len_q, d_k]
        # k: [B, n_head, len_k, d_k]
        # v: [B, n_head, len_v, d_v] where len_k == len_v
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if atten_mask is not None:
            assert atten_mask.size() == scores.size(), f"Mask size {atten_mask.size()} differs from scores size {scores.size()}"
            scores.masked_fill_(atten_mask, -1e4)
        atten = self.dropout(self.softmax(scores))
        context = torch.matmul(atten, v)
        return context, atten


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class RelMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(RelMultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale_factor = np.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.w_q = Linear(self.d_model, d_k * n_heads)
        self.w_k = Linear(self.d_model, d_k * n_heads)
        self.w_v = Linear(self.d_model, d_v * n_heads)
        self.pos_proj = Linear(self.d_model, self.d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(n_heads, d_k))
        self.v_bias = nn.Parameter(torch.Tensor(n_heads, d_k))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score

    def forward(self, x, pos_emb, atten_mask):
        batch_size = x.size(0)
        # Project inputs for queries, keys, and values
        q_ = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k)
        k_ = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v_ = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)
        
        # Project and reshape positional embeddings
        pos_emb = self.pos_proj(pos_emb).view(batch_size, -1, self.n_heads, self.d_k)

        q_ = q_ + self.u_bias
        content_score = torch.matmul((q_ + self.u_bias).transpose(1, 2), k_.transpose(2, 3))
        pos_score = torch.matmul((q_ + self.v_bias).transpose(1, 2), pos_emb.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        score = (content_score + pos_score) / self.scale_factor
        
        if atten_mask is not None:
            atten_mask = atten_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            assert atten_mask.size() == score.size(), f"Mask size {atten_mask.size()} does not match score size {score.size()}"
            score.masked_fill_(atten_mask, -1e4)
        atten = self.dropout(self.softmax(score))
        context = torch.matmul(atten, v_)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        context = self.out_proj(context)
        return context, atten


class RelMultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout, max_len, device):
        """
        :param d_model: Model dimension (e.g., 128)
        :param d_k: Key/query dimension per head (e.g., 32)
        :param d_v: Value dimension per head (e.g., 32)
        :param n_heads: Number of attention heads (e.g., 4)
        :param dropout: Dropout probability
        :param max_len: Maximum sequence length
        :param device: torch.device instance
        """
        super(RelMultiHeadAttentionLayer, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.multihead_attention = RelMultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.linear = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x, atten_mask):
        # x: [Batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        device = x.device
        pos_emb = self.positional_encoding(seq_len).to(device)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)
        residual = x
        x = self.layernorm(x)
        context, atten = self.multihead_attention(x, pos_emb, atten_mask)
        output = self.dropout(self.linear(context))
        output = output + residual
        return output, atten


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        output = self.relu(self.fc1(x))
        output = self.dropout(self.fc2(output))
        output = 0.5 * output + residual
        return output


class ConformerEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device):
        super(ConformerEncoder, self).__init__()
        self.self_attention = RelMultiHeadAttentionLayer(d_model, d_k, d_v, n_heads, dropout, max_len, device)
        self.position_wise_ff_in = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.position_wise_ff_out = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.conv_module = ConformerConvModule(in_channels=d_model)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x, atten_mask):
        # Use gradient checkpointing with use_reentrant=False
        output = checkpoint(self.position_wise_ff_in, x, use_reentrant=False)
        output, atten = checkpoint(self.self_attention, output, atten_mask, use_reentrant=False)
        output = checkpoint(self.conv_module, output, use_reentrant=False)
        output = checkpoint(self.position_wise_ff_out, output, use_reentrant=False)
        output = self.layernorm(output)
        return output, atten
