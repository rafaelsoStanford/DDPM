
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

# Padding for images to be divisible by 2^depth
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x


class SelfAttention(nn.Module):
    """
    Transformer Structure:
    
    Attention is all you need paper (https://arxiv.org/abs/1706.03762): 
        See the diagram of the transformer architecture (example: the encoder)

    1. Multihead Attention 
    2-  Normalization
    3- Feed Forward Network 
    4-  Normalization
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels        
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor):
        size = x.shape[-1] # Sequence length
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2) # View(): reshapes the tensor to the desired shape
            # -1: infer this dimension from the other given dimension; Preserve number of batches
            # swapaxes(1, 2): swap the second and third dimension -> (B, H, W) -> (B, W, H)

        x_ln = self.ln(x) # Normalize input
        attention_value, _ = self.attention(x_ln, x_ln, x_ln) #Multihead attention: Pytorch Implementation
        attention_value = attention_value + x #Add residual connection (See paper; we add the input to the output of the multihead attention)
        attention_value = self.ff_self(attention_value) + attention_value #Second residual connection (see paper)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size) #Swap back the second and third dimension and reshape to original image


class DoubleConvolution(nn.Module):
    """
    Structure taken from original UNet paper (https://arxiv.org/abs/1505.04597)
    Adjusted to fit implementation of DDPM (https://arxiv.org/abs/2006.11239) 

    Removed internal residual connections, coud not be bothered to implement them
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) #Takes inputs of (B,Cin,H,W) where B is batch size, Cin is input channels, H is height, W is width  
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)


    def forward(self, x: torch.Tensor):

        # Apply the two convolution layers and activations
        x = self.first(x)   # (B,Cin,H,W) -> (B,Cout,H,W)
        x = self.norm(x)    # Group normalization
        x = self.act(x)     # GELU activation function (https://arxiv.org/abs/1606.08415)
        x = self.second(x)  # (B,Cin,H,W) -> (B,Cout,H,W)
        return self.norm(x) # Group normalization Final output shape (B,Cout,H,W)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self, in_channels: int, out_channels: int, embeddedTime_dim=256):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, ) #2x2 max pooling windows -> reduce size of feature map by 2
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.emb_layer = nn.Sequential( # Brutally make dimensions match unsing a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim,
                out_channels
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.pool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)

        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # self.emb_layer(t) -> (B, C_out, 1, 1) 
                                                                                            #-> repeat to match image dimensions -> same time value for all pixels
        return x + emb_t
        


class UpSample(nn.Module):
    """
    ### Up-sample
    """
    def __init__(self, in_channels: int, out_channels: int, embeddedTime_dim=256):
        super().__init__()

        # Up-convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.emb_layer = nn.Sequential( # Brutally make dimensions match unsing a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim,
                out_channels
            ),
        )

    def forward(self, x: torch.Tensor, x_res: torch.Tensor, t: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, x_res], dim=1)# Concatenate along the channel dimension; kept previous feature map and upsampled feature map
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb_t
    
    
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         # (B, timesteps) -> (B, [timesteps, dim])
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim=256, device="cuda"):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.device = device

        # Define all layers used by U-net
        self.inc = DoubleConvolution(in_channels, 64)
        self.down1 = DownSample(64, 128) #set time_dim to 256 for all up and down sampling layers in init()
        self.sa1 = SelfAttention(128)
        self.down2 = DownSample(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = DownSample(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConvolution(256, 512)
        self.bot2 = DoubleConvolution(512, 512)
        self.bot3 = DoubleConvolution(512, 256)

        self.up1 = UpSample(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = UpSample(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = UpSample(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)


        # # time embeddings
        # if with_time_emb:
        #     time_dim = img_size * 4
        #     self.time_mlp = nn.Sequential(
        #         SinusoidalPositionEmbeddings(img_size),
        #         nn.Linear(img_size, time_dim),
        #         nn.GELU(),
        #         nn.Linear(time_dim, time_dim),
        #     ) # Embedding of time with image size
        # else:
        #     self.time_dim = None
        #     self.time_mlp = None

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x: torch.Tensor, t: torch.Tensor):

        # if self.time_mlp is not None:
        #     t = self.time_mlp(time) # Embed time in sinusoidal position embeddings
        # else:
        #     t = time

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Do the encoding

        # Check if the input tensor has the 2^3 divisible image size (ie downsampling 3 times)
        x, padding = pad_to(x, 2**3)

        x1 = self.inc(x)
        
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x5 = self.bot1(x4)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x3, t) # include residual connections
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        x = self.outc(x)

        x = unpad(x , padding)

        return x

