from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(self, dim, seq_len, token_mixer_dim=256, channel_mixer_dim=1024,
                 activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = Mlp(seq_len, token_mixer_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = Mlp(dim, channel_mixer_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x_t = x.transpose(1, 2)
        x_t = self.mlp_tokens(x_t)
        x_t = x_t.transpose(1, 2)
        x = x + x_t
        
        residual = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = x + residual
        return x
    

class MLPMixer_Width(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 token_mixer_dim=256, channel_mixer_dim=1024, drop_rate=0., activation='gelu'):
        super(MLPMixer_Width, self).__init__()

        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                token_mixer_dim=token_mixer_dim, channel_mixer_dim=channel_mixer_dim,
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        x = self.patchemb(images)
        
        # step 2 Go through the mixer blocks
        x = self.blocks(x)
        
        # step 3 go through layer norm
        x = self.norm(x)
        
        # step 4 Global averaging spatially
        x = x.mean(dim=1)
        
        # Classification
        x = self.head(x)
        return x
    
    def visualize(self, logdir):
        """ Visualize the token mixer layer 
        in the desired directory """
        os.makedirs(logdir, exist_ok=True)

        block0 = self.blocks[0]

        w = block0.mlp_tokens.fc1.weight.data.cpu().numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(w, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Token Mixing (Block 0) - fc1 Weight')
        plt.xlabel('Input Features (seq_len)')
        plt.ylabel('Hidden Features')

        save_path = os.path.join(logdir, 'token_mixer_block0_fc1.png')
        plt.savefig(save_path)
        plt.close()

        print(f"Visualization saved to {save_path}")
    
