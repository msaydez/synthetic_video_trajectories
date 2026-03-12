import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from mamba_ssm import Mamba

class MambaTrackletPredictor(nn.Module):
    def __init__(self, bbox_dim=4, latent_dim=64,
                 mlp_hidden_dim=64, trajectory_embedding_dim=64):
        super().__init__()

        # Linear projection: bbox (x,y,w,h) -> latent_dim
        self.bbox_encoder = nn.Linear(bbox_dim, latent_dim)


        self.mamba1 = Mamba(d_model=latent_dim, d_state=16, d_conv=4, expand=2)

        self.mamba2 = Mamba(d_model=latent_dim, d_state=16, d_conv=4, expand=2)

        # Head 1: bbox delta prediction (motion regression)
        self.mlp_delta = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, bbox_dim)
        )

        # Head 2: trajectory embedding output (for association / matching)
        self.mlp_traj_embed = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, trajectory_embedding_dim),
            nn.LayerNorm(trajectory_embedding_dim),  # normalize embedding
        )

    def forward(self, bbox_sequence):
        """
        bbox_sequence: Tensor of shape (batch_size, seq_len, 4)
        """
        # Step 1: Encode input
        latent_sequence = self.bbox_encoder(bbox_sequence)  # (B, L, latent_dim)

        # Step 2: Process motion history with Mamba
        mamba_output = self.mamba1(latent_sequence)  # (B, L, latent_dim)
        mamba_output = self.mamba2(mamba_output)  # (B, L, latent_dim)

        # Step 3: Take last timestep only
        last_hidden = mamba_output[:, -1, :]  # (B, latent_dim)

        # Step 4a: Predict bbox delta
        bbox_delta = self.mlp_delta(last_hidden)  # (B, 4)

        # Step 4b: Generate trajectory embedding
        traj_embed = self.mlp_traj_embed(last_hidden)  # (B, trajectory_embedding_dim)

        return bbox_delta, traj_embed
