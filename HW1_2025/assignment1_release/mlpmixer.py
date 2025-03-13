from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Use convolution to perform patch embedding.
        # This converts an input of shape [B, 3, img_size, img_size] into
        # [B, embed_dim, grid_size, grid_size]. Flattening then gives [B, num_patches, embed_dim].
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return: tensor of shape [batch, num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # Convert from BCHW to BNC (batch, num_patches, embed_dim)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MixerBlock(nn.Module):
    """Residual Block with token mixing and channel MLPs.
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        
        # For token mixing, the MLP operates on the tokens dimension.
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        
        # For channel mixing, the MLP operates on the channel dimension.
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Token mixing
        y = self.norm1(x)            # [B, N, C]
        y = y.transpose(1, 2)         # [B, C, N] - now tokens become the feature dimension for the MLP
        y = self.mlp_tokens(y)        # MLP applied on token dimension
        y = y.transpose(1, 2)         # [B, N, C]
        x = x + y                   # Residual connection
        
        # Channel mixing
        z = self.norm2(x)
        z = self.mlp_channels(z)
        out = x + z                 # Residual connection
        return out

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(dim=embed_dim, seq_len=self.patchemb.num_patches, activation=activation, drop=drop_rate)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """
        :param images: tensor of shape [batch, 3, img_size, img_size]
        :return: logits of shape [batch, num_classes]
        """
        # Step 1: Patch embedding
        x = self.patchemb(images)    # [B, num_patches, embed_dim]
        # Step 2: Mixer blocks
        x = self.blocks(x)
        # Step 3: Layer normalization
        x = self.norm(x)
        # Step 4: Global average pooling over the token dimension
        x = x.mean(dim=1)
        # Step 5: Classification head
        x = self.head(x)
        return x
    
    def visualize(self, logdir):
        """
        Visualize the token mixer layer (first FC layer in token-mixing MLP) from the first MixerBlock.
        Saves a heatmap of the weights to the specified log directory.
        """
        # Extract weights from the first token-mixing FC layer.
        token_mixer_weights = self.blocks[0].mlp_tokens.fc1.weight.data.cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(token_mixer_weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Token Mixer FC1 Weights')
        os.makedirs(logdir, exist_ok=True)
        plt.savefig(os.path.join(logdir, 'token_mixer_fc1_weights.png'))
        plt.close()

    
