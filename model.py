import torch
import torch.nn as nn
import torchvision.transsforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.sequentail(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True ),
            nn.conv2d(out_channels, out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512] #initializing the features
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() #to do module.eval 
        self.downs = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Down Part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = features

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2
                )
            )
