import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, L, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [B, L, D]
        x = x + self.pe[:, :x.size(1), :]
        return x

class ChromaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mark_specific_context = getattr(config, "USE_MARK_SPECIFIC_CONTEXT", False)
        
        # 1. DNA Encoder (CNN Stem)
        self.cnn = nn.Sequential(
            nn.Conv1d(config.CNN_CHANNELS[0], config.CNN_CHANNELS[1], config.CNN_KERNELS[0], padding=config.CNN_KERNELS[0]//2),
            nn.BatchNorm1d(config.CNN_CHANNELS[1]),
            nn.GELU(),
            nn.Conv1d(config.CNN_CHANNELS[1], config.CNN_CHANNELS[2], config.CNN_KERNELS[1], stride=config.CNN_STRIDES[1], padding=config.CNN_KERNELS[1]//2),
            nn.BatchNorm1d(config.CNN_CHANNELS[2]),
            nn.GELU(),
            nn.Conv1d(config.CNN_CHANNELS[2], config.CNN_CHANNELS[3], config.CNN_KERNELS[2], stride=config.CNN_STRIDES[2], padding=config.CNN_KERNELS[2]//2),
            nn.BatchNorm1d(config.CNN_CHANNELS[3]),
            nn.GELU()
        )
        
        # 1b. Dilated CNN Block
        if getattr(config, "USE_DILATED_CONVS", False):
            layers = []
            in_channels = config.CNN_CHANNELS[3]
            for d in getattr(config, "CNN_DILATIONS", [2, 4, 8]):
                layers.extend([
                    nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=d, dilation=d),
                    nn.BatchNorm1d(in_channels),
                    nn.GELU()
                ])
            self.dilated_block = nn.Sequential(*layers)
        else:
            self.dilated_block = None
        
        # DNA Transformer
        self.pos_encoder = PositionalEncoding(config.D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_MODEL * 4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)
        
        # 2. RNA Encoder (MLP)
        self.rna_mlp = nn.Sequential(
            nn.Linear(config.RNA_INPUT_DIM, 512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(512, config.RNA_HIDDEN_DIM)
        )
        
        # 3. FiLM Fusion Generators
        self.film_gamma = nn.Linear(config.RNA_HIDDEN_DIM, config.D_MODEL)
        self.film_beta = nn.Linear(config.RNA_HIDDEN_DIM, config.D_MODEL)
        
        # 4. Attention Pooling
        self.attn_query = nn.Parameter(torch.randn(1, 1, config.D_MODEL))
        self.pool_attn = nn.MultiheadAttention(config.D_MODEL, num_heads=1, batch_first=True)
        if self.use_mark_specific_context:
            self.mark_queries = nn.Parameter(torch.randn(config.NUM_MARKS, 1, config.D_MODEL))
        else:
            self.mark_queries = None
        
        # 5. Multitask Regression Heads
        # Shared layer after fusion
        self.shared_head = nn.Sequential(
            nn.Linear(config.D_MODEL + config.RNA_HIDDEN_DIM, config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # One parallel regression head per target mark
        self.mark_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.HEAD_HIDDEN_DIM, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(config.NUM_MARKS)
        ])

    def _context_token_bounds(self, mark_name, seq_len):
        context_bp = getattr(config, "MARK_CONTEXT_BP", {}).get(mark_name, config.WINDOW_SIZE)
        context_bp = max(1, min(int(context_bp), int(config.WINDOW_SIZE)))
        context_tokens = max(1, int(round(seq_len * context_bp / config.WINDOW_SIZE)))

        center = seq_len // 2
        half = context_tokens // 2
        start = max(0, center - half)
        end = min(seq_len, start + context_tokens)
        start = max(0, end - context_tokens)
        return start, end

    def _pool_mark_features(self, dna_cond):
        batch_size, seq_len, _ = dna_cond.shape
        pooled = []

        for m_idx, mark_name in enumerate(config.TARGET_MARKS):
            start, end = self._context_token_bounds(mark_name, seq_len)
            mark_tokens = dna_cond[:, start:end, :]
            q = self.mark_queries[m_idx].unsqueeze(0).expand(batch_size, -1, -1)
            attn_out, _ = self.pool_attn(query=q, key=mark_tokens, value=mark_tokens)
            pooled.append(attn_out.squeeze(1))

        return pooled

    def forward(self, dna_seq, rna_feat):
        # dna_seq: [B, 4, L]
        # rna_feat: [B, RNA_INPUT_DIM]
        B = dna_seq.size(0)
        
        # --- RNA Branch ---
        rna_emb = self.rna_mlp(rna_feat) # [B, RNA_HIDDEN_DIM]
        
        # Generate FiLM conditioning parameters
        gamma = self.film_gamma(rna_emb).unsqueeze(1) # [B, 1, D_MODEL]
        beta = self.film_beta(rna_emb).unsqueeze(1)   # [B, 1, D_MODEL]
        
        # --- DNA Branch ---
        dna_feat = self.cnn(dna_seq) # [B, D_MODEL, L/16]
        
        if self.dilated_block is not None:
            # Residual connection
            dna_feat = dna_feat + self.dilated_block(dna_feat)
            
        dna_feat = dna_feat.transpose(1, 2) # [B, L/16, D_MODEL]
        
        dna_feat = self.pos_encoder(dna_feat)
        dna_feat = self.transformer(dna_feat) # [B, L/16, D_MODEL]
        
        # --- FiLM Fusion ---
        # Modulate DNA features with RNA context
        dna_cond = (1 + gamma) * dna_feat + beta # [B, L/16, D_MODEL]
        
        # --- Multitask Heads ---
        outputs = []
        if self.use_mark_specific_context:
            pooled_feats = self._pool_mark_features(dna_cond)
            for mark_feat, head in zip(pooled_feats, self.mark_heads):
                combined = torch.cat([mark_feat, rna_emb], dim=1)
                shared_feat = self.shared_head(combined)
                outputs.append(head(shared_feat))
        else:
            # Query: [B, 1, D]
            q = self.attn_query.expand(B, -1, -1)
            attn_out, _ = self.pool_attn(query=q, key=dna_cond, value=dna_cond)
            pool_feat = attn_out.squeeze(1) # [B, D_MODEL]

            combined = torch.cat([pool_feat, rna_emb], dim=1) # [B, D_MODEL + RNA_HIDDEN_DIM]
            shared_feat = self.shared_head(combined) # [B, HEAD_HIDDEN_DIM]
            for head in self.mark_heads:
                outputs.append(head(shared_feat))
            
        # Concat along last dim
        out = torch.cat(outputs, dim=1) # [B, NUM_MARKS]
        
        return out
