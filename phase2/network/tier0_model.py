"""
tier0_model.py -- the network. Four steps, read top to bottom in forward():

  1. Project each past event (a row of numbers) and the current filing into
     vectors of size d_model.
  2. Put a learned SUMMARY token in front of the company's history and run a
     small Transformer encoder. Through self-attention the summary token reads
     every past event and learns which ones matter. (Same idea as BERT's [CLS].)
     It is never padding, so even an empty history has one valid position.
  3. Join the summary with the current filing's vector.
  4. Two heads: predicted CAR[-1,+1] (regression) and reaction_category (5 classes).

Step 2 has two versions, chosen by causal_readout (tier0_config.CAUSAL_READOUT):
  False -- no mask; the summary token at position 0 is read out.
  True  -- causal mask (each position sees only itself and earlier ones); the last
           real position is read out, and position 0 is just a start token that is
           read out only when the history is empty.

Time order needs no special machinery: every past event carries its own
"trading days ago" and "days since previous filing" columns (built in
tier0_train.history_for), and nothing in the history is in the future.
With USE_TIME_CYCLES on (tier0_config.py), each past event also carries
sin/cos of trading-days-ago at weekly/monthly/quarterly/yearly periods --
an ablation for whether cyclical timing helps.
"""
import torch
import torch.nn as nn


class Tier0Model(nn.Module):
    def __init__(self, n_history_features, n_current_features, n_classes,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1, causal_readout=False):
        super().__init__()
        self.causal_readout = causal_readout     # which history summary to use (see tier0_config.CAUSAL_READOUT)
        self.history_proj = nn.Linear(n_history_features, d_model)
        self.current_proj = nn.Linear(n_current_features, d_model)
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.combine = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.car_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, n_classes)

    def forward(self, history, history_len, current):
        """history (B, L, n_history_features), padded with zeros after history_len rows;
        history_len (B,); current (B, n_current_features)."""
        B, L, _ = history.shape

        # 1-2. summary token + projected history, padding hidden from attention
        seq = torch.cat([self.summary_token.expand(B, 1, -1), self.history_proj(history)], dim=1)
        position = torch.arange(L + 1, device=history.device)
        is_padding = position[None, :] > history_len[:, None]      # position 0 (summary) is never padding

        if not self.causal_readout:
            # Design A: everyone attends to everyone; read out the summary token at position 0.
            summary = self.encoder(seq, src_key_padding_mask=is_padding)[:, 0]
        else:
            # Design B: causal mask -- row = the position looking, column = the position looked at;
            # True above the diagonal blocks every look at a LATER position.
            looks_ahead = torch.triu(torch.ones(L + 1, L + 1, dtype=torch.bool, device=history.device), diagonal=1)
            encoded = self.encoder(seq, mask=looks_ahead, src_key_padding_mask=is_padding)
            # Read out the last real position: true_len - 1 = (history_len + 1) - 1 = history_len.
            # With an empty history that is position 0, the start token.
            summary = encoded[torch.arange(B, device=history.device), history_len]

        # 3-4. combine with the current filing, then predict
        hidden = self.combine(torch.cat([summary, self.current_proj(current)], dim=-1))
        return self.car_head(hidden).squeeze(-1), self.class_head(hidden)
