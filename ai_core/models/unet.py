import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        e4 = self.d4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.c4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.c3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.c2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.c1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)