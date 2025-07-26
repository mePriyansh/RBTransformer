import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin


#####################################################################################################
#                                    BDE Projection Layer Class                                     #
#####################################################################################################
class BDEProjectionLayer(nn.Module):
    """
    Projects Band Differential Entropy (BDE) tokens into a higher-dimensional embedding space.

    Args:
        bde_dim (int): Dimension of BDE features.
        embed_dim (int): Output embedding dimension.
    """

    def __init__(self, bde_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(bde_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Shape (batch_size, num_electrodes, bde_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return self.linear(x)


#####################################################################################################
#                             Electrode Identity Embedding Class                                    #
#####################################################################################################
class ElectrodeIdentityEmbedding(nn.Module):
    """
    Adds a learnable identity embedding to each electrode position via element-wise addition.

    Args:
        num_electrodes (int): Number of EEG electrodes.
        embed_dim (int): Output embedding dimension.
    """

    def __init__(self, num_electrodes, embed_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, num_electrodes, embed_dim))

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return x + self.embedding


#####################################################################################################
#                                  Classification-Head Class                                        #
#####################################################################################################
class ClassificationHead(nn.Module):
    """
    Final classification layer for affective state prediction.

    Args:
        embed_dim (int): Output embedding dimension.
        num_classes (int): Number of affective classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, num_classes, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_classes)
        """
        x = x.mean(dim=1)  # global average pooling over electrodes
        x = self.norm(x)
        return self.mlp(x)


#####################################################################################################
#                                          RBTransformer                                            #
#####################################################################################################
class RBTransformer(nn.Module, PyTorchModelHubMixin):
    """
    Final end-to-end RBTransformer model for EEG-based affective state classification.

    Args:
        num_electrodes (int): Number of EEG electrodes.
        bde_dim (int): Dimension of BDE features.
        embed_dim (int): Output embedding dimension.
        depth (int): Number of stacked InterCorticalAttentionBlock layers.
        heads (int): Number of attention heads per block.
        head_dim (int): Embedding dimension per attention head.
        mlp_hidden_dim (int): Hidden layer size in feed-forward networks.
        dropout (float): Dropout probability.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        num_electrodes=14,
        bde_dim=4,
        embed_dim=64,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        # Configure and initialize model layers of RBTransformer
        self.bde_proj_layer = BDEProjectionLayer(bde_dim, embed_dim)
        self.electrode_id_embedding = ElectrodeIdentityEmbedding(
            num_electrodes, embed_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.classification_head = ClassificationHead(embed_dim, num_classes, dropout)

        # Save model configuration for reproducibility
        self.config = {
            "num_electrodes": num_electrodes,
            "bde_dim": bde_dim,
            "embed_dim": embed_dim,
            "dropout": dropout,
            "num_classes": num_classes,
        }

    def forward(self, x):
        """
        Forward pass of the RBTransformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_electrodes, bde_dim)

        Returns:
            Tensor: Logits of shape (batch_size, num_classes)
        """
        # Flatten extra dimension if present (e.g., from dataloader batch shapes)
        if x.dim() > 3:
            x = x.view(x.size(0), -1, self.config["bde_dim"])

        # Project BDE features into embedding space
        x = self.bde_proj_layer(x)

        # Add learnable electrode identity embeddings
        x = self.electrode_id_embedding(x)

        # Apply dropout post-embedding
        x = self.dropout(x)

        # Final classification using deep MLP head
        return self.classification_head(x)