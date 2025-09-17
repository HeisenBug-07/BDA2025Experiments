import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import *
import conformer as cf

def truncate_tensor(x, atten_mask, max_len):
    """Truncate tensor and corresponding attention mask to max_len along time dimension."""
    if x.size(1) > max_len:
        x = x[:, :max_len, :]
        atten_mask = atten_mask[:, :max_len, :max_len]
    return x, atten_mask

class Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim=392, feat_dim=64,
                 d_k=64, d_v=64, d_ff=2048, n_heads=8,
                 max_attn_len=512,  # new parameter for truncation
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Transformer_E2E_LID, self).__init__()
        
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.max_attn_len = max_attn_len
        
        # Input processing
        self.transform = nn.Linear(input_dim, feat_dim)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, 
                                               features_dim=feat_dim, 
                                               device=device)
        
        # Transformer configuration
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([
            EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout)
            for _ in range(4)
        ])
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_lang)
        )

    def forward(self, x, seq_len, atten_mask):
        # Input validation
        batch_size, seq_length, input_dim = x.size()
        if input_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
        
        # Input processing
        x = self.transform(x)  # [B, T, feat_dim]
        x = self.layernorm1(x)
        x = self.pos_encoding(x, seq_len)
        
        # Truncate if necessary to reduce memory usage
        x, atten_mask = truncate_tensor(x, atten_mask, self.max_attn_len)
        
        # Prepare multi-head attention
        x = x.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [B, n_heads, T, feat_dim]
        x = x.transpose(1, 2).contiguous()  # [B, T, n_heads, feat_dim]
        x = x.view(batch_size, -1, self.d_model)  # [B, T, d_model]
        
        # Process through transformer blocks
        for block in self.blocks:
            x, _ = block(x, atten_mask)
        
        # Statistics pooling
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return self.fc(stats)

class X_Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim=392, feat_dim=64,
                 d_k=64, d_v=64, d_ff=2048, n_heads=4,
                 dropout=0.1, n_lang=12, max_seq_len=10000,
                 max_attn_len=512,  # new parameter for truncation
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(X_Transformer_E2E_LID, self).__init__()
        
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.max_attn_len = max_attn_len
        
        # TDNN layers for 392-dim input
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, dilation=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, 5, dilation=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1, dilation=1),
            nn.BatchNorm1d(512)
        )
        
        # Feature processing
        self.fc_xv = nn.Linear(1024, feat_dim)
        self.layernorm = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len,
                                               features_dim=feat_dim,
                                               device=device)
        
        # Transformer configuration
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([
            EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout)
            for _ in range(2)
        ])
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_lang)
        )

    def forward(self, x, seq_len, atten_mask):
        # Input validation
        batch_size, seq_length, input_dim = x.size()
        if input_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
        
        # TDNN processing requires [B, C, T]
        x = x.transpose(1, 2)  # [B, 392, T]
        x = self.tdnn(x)  # [B, 512, T]
        
        # Statistics pooling
        stats = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)  # [B, 1024]
        x = self.fc_xv(stats)  # [B, feat_dim]
        
        # Prepare for transformer: expand to sequence length dimension
        x = x.unsqueeze(1).expand(-1, seq_length, -1)  # [B, T, feat_dim]
        x = self.layernorm(x)
        x = self.pos_encoding(x, seq_len)
        
        # Truncate if necessary to reduce memory usage
        x, atten_mask = truncate_tensor(x, atten_mask, self.max_attn_len)
        
        # Multi-head attention preparation
        x = x.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [B, n_heads, T, feat_dim]
        x = x.transpose(1, 2).contiguous()  # [B, T, n_heads, feat_dim]
        x = x.view(batch_size, -1, self.d_model)  # [B, T, d_model]
        
        # Process through transformer blocks
        for block in self.blocks:
            x, _ = block(x, atten_mask)
            
        # Final prediction
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return self.fc(stats)

class Conformer(nn.Module):
    def __init__(self, input_dim=392, feat_dim=64,
                 d_k=64, d_v=64, n_heads=8, d_ff=2048,
                 max_len=10000, dropout=0.1, n_lang=14,
                 max_attn_len=512,  # new parameter for truncation
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Conformer, self).__init__()
        
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.max_attn_len = max_attn_len
        
        # Input processing
        self.conv_subsample = cf.Conv1dSubsampling(input_dim, feat_dim)
        self.transform = nn.Linear(feat_dim, feat_dim)
        self.layernorm = LayerNorm(feat_dim)
        
        # Conformer blocks
        self.d_model = feat_dim * n_heads
        self.blocks = nn.ModuleList([
            cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
            for _ in range(4)
        ])
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_lang)
        )

    def forward(self, x, atten_mask):
        # Input validation
        batch_size, seq_length, input_dim = x.size()
        if input_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
        
        # Feature extraction
        x = self.conv_subsample(x)  # [B, T, feat_dim]
        x = self.transform(x)
        x = self.layernorm(x)
        
        # Truncate if necessary
        x, atten_mask = truncate_tensor(x, atten_mask, self.max_attn_len)
        
        # Multi-head attention preparation
        x = x.unsqueeze(1).repeat(1, self.blocks[0].n_heads, 1, 1)  # [B, n_heads, T, feat_dim]
        x = x.transpose(1, 2).contiguous()  # [B, T, n_heads, feat_dim]
        x = x.view(batch_size, -1, self.d_model)  # [B, T, d_model]
        
        # Process through conformer blocks
        for block in self.blocks:
            x, _ = block(x, atten_mask)
            
        # Statistics pooling
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return self.fc(stats)
