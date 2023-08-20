import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, image_channels, down_channels, up_channels, out_dim, time_emb_dim, context_dim):
        super(Unet, self).__init__()
        self.image_channels = image_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.out_dim = image_channels
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim

        self.pos_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Initial convolutional layer
        self.init_conv =  nn.Sequential(
            nn.Conv2d(image_channels, self.down_channels[0], 3, padding=1), # same size conv
            nn.BatchNorm2d(self.down_channels[0]),
            nn.GELU()
            )

        # Downsample
        self.downs =  nn.ModuleList([
            ResnetBlockDown(down_channels[i], down_channels[i+1]) for i in range(len(down_channels)-1)
            ])
        
        # Bottleneck layer
        self.middle_block = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU(),
            nn.ConvTranspose2d(down_channels[-1], up_channels[0], 4, 4),
            nn.GroupNorm(8, up_channels[0]),
            nn.ReLU()
            )
        
        # Upsample
        self.ups = nn.ModuleList([
            ResnetBlockUp(up_channels[i], up_channels[i+1]) for i in range(len(up_channels)-1)
            ])

        # Context embedding layer
        self.context_embs = nn.ModuleList([
            SimpleEmbed(context_dim, up_channels[i]) for i in range(len(up_channels)-1)
            ])
        
        # Time embedding layer
        self.time_embs = nn.ModuleList([
            SimpleEmbed(self.time_emb_dim, up_channels[i]) for i in range(len(up_channels)-1)
            ])
        
        # Final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(up_channels[-1]*2, up_channels[-1], 3, 1, 1), # reduce number of feature maps
            nn.GroupNorm(8, up_channels[-1]), # normalize
            nn.ReLU(),
            nn.Conv2d(up_channels[-1], self.image_channels, 3, 1, 1), # map to same number of channels as input
            )
        
    def forward(self, x, t, c):
        """
        x : (batch, channel, h, w) : input image
        t : (batch, 1) : time step
        c : (batch, context_dim) : context
        """
        
        x = self.init_conv(x)
        init_x = torch.clone(x)        
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)
        x = self.middle_block(x)
        pos = self.pos_emb(t)
        for i in range(len(self.ups)):
            up = self.ups[i]
            c_emb = self.context_embs[i]
            t_emb = self.time_embs[i]
            residual_x = residual_inputs.pop()
            t_ = t_emb(pos).view(-1, t_emb.emb_dim, 1, 1)
            c_ = c_emb(c).view(-1, c_emb.emb_dim, 1, 1)
            # Add residual x as additional channels
            x = up(x*c_+t_, residual_x)
        out = self.out(torch.cat((x, init_x),1))
        return out

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.view(-1,1) * embeddings.view(1,-1)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleEmbed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(SimpleEmbed, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)    

class ResnetBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlockDown, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # same size conv
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU()   # GELU activation function
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # same size conv
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU()   # GELU activation function
        )
        self.transform = nn.Conv2d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.max_pool(x2)
    
class ResnetBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlockUp, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels*2, out_channels, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # same size conv
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU()   # GELU activation function
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # same size conv
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU()   # GELU activation function
        )
        self.transform = nn.Conv2d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
    
    def forward(self, x, residual):
        x = torch.cat((x, residual), 1)
        xt = self.up_conv(x)
        x1 = self.conv1(xt)
        x2 = self.conv2(x1)
        return x2
