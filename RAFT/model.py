import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrievalModule(nn.Module):
    def __init__(self, d_model, top_k):
        super(RetrievalModule, self).__init__()
        self.d_model = d_model
        self.top_k = top_k

    def forward(self, query, keys, values, reverse=False):
        """
        query: [B, C, d_model]
        keys: [M, d_model] - Memory Keys
        values: [M, d_model] - Memory Values (Patches)
        reverse: bool - Whether to use Reverse Similarity
        """
        B, C, d = query.shape
        M, _ = keys.shape

        # Calculate Scores: Q . K^T
        # query: (B, C, d)
        # keys: (M, d) -> transpose (d, M)
        # scores: (B, C, M)
        scores = torch.matmul(query, keys.transpose(-1, -2))

        # Scaling (optional but standard)
        scale = self.d_model ** -0.5
        scores = scores * scale

        if reverse:
            # For Reverse Mode, we want lowest Q.K^T (most negative).
            # Negating scores effectively makes the most negative ones the highest positive ones.
            scores = -scores

        # Select Top-k
        # topk_scores: (B, C, k)
        # topk_indices: (B, C, k)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)

        # Softmax to get Attention Weights
        attn_weights = F.softmax(topk_scores, dim=-1) # (B, C, k)

        # Retrieve Values
        # values: (M, d)
        # We need to gather values based on topk_indices.
        # topk_indices is (B, C, k). Values is (M, d).
        # F.embedding handles (..., ) indices and (M, d) weight
        retrieved_values = F.embedding(topk_indices, values) # (B, C, k, d)

        if reverse:
            # Value Handling: Flip sign for Reverse Mode
            retrieved_values = -retrieved_values

        # Aggregation (Weighted Sum)
        # attn_weights: (B, C, k) -> (B, C, k, 1)
        # retrieved_values: (B, C, k, d)
        # aggregated: (B, C, d)
        aggregated = torch.sum(retrieved_values * attn_weights.unsqueeze(-1), dim=2)

        return aggregated

class RAFT(nn.Module):
    def __init__(self, enc_in, d_model, top_k, seq_len, pred_len):
        super(RAFT, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.enc_in = enc_in

        # 1. Downsampling (Input Projection)
        # Projects input sequence length to feature dimension F (d_model)
        # Input: (B, C, L) -> Output: (B, C, d_model)
        self.downsample = nn.Linear(seq_len, d_model)

        # 2. Dual Retrieval
        self.retrieval = RetrievalModule(d_model, top_k)

        # 4. Intermediate Fusion (Mixer)
        # Normal Feature: Concat [Projected_Input, Aggregated_Normal] -> MLP -> (B, C, F)
        self.normal_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Reverse Feature: Concat [Projected_Input, Aggregated_Reverse] -> MLP -> (B, C, F)
        self.reverse_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 5. Final Fusion
        # Concat [Normal_Feature, Reverse_Feature] -> (B, C, 2F) -> Linear -> (B, C, F)
        self.final_fusion = nn.Linear(2 * d_model, d_model)

        # 6. Prediction
        # (B, C, F) -> (B, C, pred_len)
        self.prediction = nn.Linear(d_model, pred_len)

    def forward(self, x, keys, values):
        """
        x: [B, seq_len, C] - Input History
        keys: [M, d_model] - Memory Keys (Patches)
        values: [M, d_model] - Memory Values (Patches/Targets)
        """
        # Transpose x to [B, C, seq_len] for channel-independent processing
        x = x.permute(0, 2, 1) # (B, C, L)

        # 1. Downsampling
        projected_input = self.downsample(x) # (B, C, F)

        # 2. Dual Retrieval
        # Stream A (Normal)
        # aggregated_normal: (B, C, F)
        aggregated_normal = self.retrieval(projected_input, keys, values, reverse=False)

        # Stream B (Reverse)
        # aggregated_reverse: (B, C, F)
        aggregated_reverse = self.retrieval(projected_input, keys, values, reverse=True)

        # 4. Intermediate Fusion (The "Mixer" Step)
        # Normal Feature
        normal_concat = torch.cat([projected_input, aggregated_normal], dim=-1) # (B, C, 2F)
        normal_feature = self.normal_mlp(normal_concat) # (B, C, F)

        # Reverse Feature
        reverse_concat = torch.cat([projected_input, aggregated_reverse], dim=-1) # (B, C, 2F)
        reverse_feature = self.reverse_mlp(reverse_concat) # (B, C, F)

        # 5. Final Fusion
        fusion_concat = torch.cat([normal_feature, reverse_feature], dim=-1) # (B, C, 2F)
        fused_embedding = self.final_fusion(fusion_concat) # (B, C, F)

        # 6. Prediction
        out = self.prediction(fused_embedding) # (B, C, pred_len)

        # Transpose back to [B, pred_len, C]
        out = out.permute(0, 2, 1)

        return out
