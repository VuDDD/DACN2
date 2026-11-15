import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.unet import DoubleConv, Down, OutConv

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) mechanism proposed by Oktay et al. (2018).
    Selectively weights spatial features coming from encoder skip-connections
    conditioned on coarse contextual gating signals from the deeper decoder stage.
    
    This suppresses irrelevant background retinal regions and highlights subtle
    focal lesions such as Microaneurysms (MA) and fine capillary blood vessels.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: number of channels in gating signal (decoder)
            F_l: number of channels in skip connection (encoder)
            F_int: number of intermediate channels for cross-attention
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from deeper decoder layer [B, F_g, H_g, W_g]
            x: Skip-connection feature map from encoder [B, F_l, H_l, W_l]
        Returns:
            Attention-weighted feature map [B, F_l, H_l, W_l]
        """
        # Align spatial dimension of g to x if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        net = self.relu(g1 + x1)
        psi = self.psi(net)
        return x * psi


class AttentionUpBlock(nn.Module):
    """
    Decoder Up block integrated with an Attention Gate on the skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_dec)
        # Pad if odd dimensions
        diffY = x_enc.size()[2] - x_up.size()[2]
        diffX = x_enc.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Apply Attention Gate to the encoder skip connection
        x_enc_att = self.ag(g=x_up, x=x_enc)
        
        # Concatenate along channel dimension
        x = torch.cat([x_enc_att, x_up], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net Architecture.
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 5):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder Path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder Path with Attention Gates
        self.up1 = AttentionUpBlock(1024, 512)
        self.up2 = AttentionUpBlock(512, 256)
        self.up3 = AttentionUpBlock(256, 128)
        self.up4 = AttentionUpBlock(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits [B, n_classes, H, W]."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions [B, n_classes, H, W] after sigmoid activation."""
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

