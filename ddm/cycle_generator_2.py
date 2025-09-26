import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch import einsum
from functools import partial
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch.nn import Module
#################################################################### Cycle GAN ##########################################################################

def init_func(net):  # define the initialization function
    for m in net.modules():
        init_gain = 0.02
        init_type = 'normal'
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlock_time(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_layer, use_dropout, use_bias, time_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_time, self).__init__()
        self.build_conv_block(dim, norm_layer, use_dropout, use_bias, time_dim)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias, time_dim):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        block_klass = partial(ResnetBlock, groups=8)

        self.conv_block1 = nn.Sequential(
            Conv2dBlock(dim, dim, kernel_size=3, stride=1, padding=1, norm='inst', activation='relu', pad_type='reflect')
        )
        
        if use_dropout:
            self.conv_block1.append(nn.Dropout(0.5))

        self.conv_block2 = nn.Sequential(
            Conv2dBlock(dim, dim, kernel_size=3, stride=1, padding=1, norm='inst', activation='none', pad_type='reflect')
        )

        self.block_klass1 = block_klass(dim, dim, time_emb_dim=time_dim)
        self.block_klass2 = block_klass(dim, dim, time_emb_dim=time_dim)

        # self.conv_block1.apply(init_func)
        # self.conv_block2.apply(init_func)

    def forward(self, x, t):
        """Forward function (with skip connections)"""
        origin_input = x
        mid_input = self.conv_block1(self.block_klass1(x, t))
        out = origin_input + self.conv_block2(self.block_klass2(mid_input, t))  # add skip connections
        return out

## build model for generator-gan
def exists(x):
    return x is not None

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)    

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock_attn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, heads = 2, dim_head = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.linear_attn = LinearAttention(dim_out, heads = heads, dim_head = dim_head, num_mem_kv = 2)
        # self.attention = nn.MultiheadAttention(embed_dim=dim_out, num_heads=2, batch_first=True)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        if exists(self.mlp) and exists(time_emb):
            h = self.linear_attn(h)
        # if exists(self.mlp) and exists(time_emb):
        #     B, C, H, W = h.shape
        #     h = h.view(B, C, H*W).permute(0, 2, 1)
        #     h = self.attention(h, h, h)[0]
        #     h = h.permute(1, 2, 0).contiguous().view(B, C, H, W)

        return h + self.res_conv(x)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True) if use_act else nn.Identity()
        )
        # self.conv.apply(init_func)
    def forward(self, x):
        return self.conv(x)

class ResnetGenerator_timestep_restime_2_attn(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, 
                norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
                use_dropout=False, 
                n_blocks=9, 
                dim=128,
                learned_sinusoidal_dim = 16,
                random_fourier_features = False,
                learned_sinusoidal_cond = False,
                resnet_block_groups=8,
                ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_timestep_restime_2_attn, self).__init__()
        use_bias = True
        
        # block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        block_klass = partial(ResnetBlock_attn, groups = resnet_block_groups)

        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # init_conv
        self.init_conv = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True))
        
        # down_blocks
        self.downsample = nn.ModuleList([])
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            self.downsample.append(nn.ModuleList([
                block_klass(ngf * mult, ngf * mult, time_emb_dim=time_dim),
                block_klass(ngf * mult, ngf * mult, time_emb_dim=time_dim),
                ConvBlock(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                ]))

        # residual_blocks
        mult = 2 ** n_downsampling
        self.residual_blocks = nn.ModuleList([
            ResnetBlock_time(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, time_dim=time_dim) for _ in range(n_blocks)
        ])
        # up_blocks
        self.upsample = nn.ModuleList([])
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            self.upsample.append(nn.ModuleList([
                block_klass(ngf * mult, ngf * mult, time_emb_dim=time_dim),
                block_klass(ngf * mult, ngf * mult, time_emb_dim=time_dim),
                ConvBlock(ngf * mult, int(ngf * mult / 2), down=False, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                ]))

        self.last = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        # self.init_conv.apply(init_func)
        # self.last.apply(init_func)

    def forward(self, input, time):
        """Standard forward"""
        t = self.time_mlp(time)
        x = self.init_conv(input)
        
        for block1, block2, convblock in self.downsample:
            x = block1(x, t)
            x = block2(x, t)

            x = convblock(x)
        
        for block in self.residual_blocks:
            x = block(x, t)
        
        for block1, block2, convblock in self.upsample:
            x = block1(x, t)
            x = block2(x, t)

            x = convblock(x)

        return self.last(x)

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=False)
    # nn.InstanceNorm2d(out_channels),

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        
    
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x):
        # x: b, h, w, d
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        pos = torch.cat((pos_y, pos_x), dim=3).contiguous()
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h, w, d
        '''
        return pos

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CondAttention(nn.Module): # Multi-Scale Window-Attention
    def __init__(self, dim, dim2, hidden_dim, heads=4, window_size_q=[4, 4],
                 window_size_k=[[4, 4], [2, 2], [1, 1]], drop=0.1):
        super(CondAttention, self).__init__()
        # assert  dim == heads * dim_head
        dim_head = hidden_dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        # hidden_dim = dim_head * heads
        # self.qkv = nn.Conv3d(dim, hidden_dim*3, 1)
        self.q_lin = nn.Linear(dim, hidden_dim)
        self.k_lin = nn.Linear(dim2, hidden_dim)
        self.v_lin = nn.Linear(dim2, hidden_dim)
        self.pos_enc = PositionEmbeddingSine(hidden_dim)
        self.window_size_q = window_size_q
        self.avgpool_q = nn.AdaptiveAvgPool2d(output_size=window_size_q)
        # self.avgpool_ks = nn.ModuleList()
        # for i in range(len(window_size_k)):
        #     self.avgpool_ks.append(nn.AdaptiveAvgPool2d(output_size=window_size_k[i]))
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim*2, out_features=dim, drop=drop)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x, cond):
        # x: B, C, H, W
        # cond: B, C
        B, C, H, W = x.shape
        shortcut = x
        q_s = self.avgpool_q(x)
        length = q_s.shape[-2] * q_s.shape[-1]
        kg = cond.unsqueeze(1)#.expand(-1, length, -1)
        qg = self.avgpool_q(x).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C)


        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        qg = self.q_lin(qg).reshape(B, num_window_q, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                            3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg = kg2
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C)
        qg = qg.transpose(1, 2).reshape(B, C, self.window_size_q[0], self.window_size_q[1])
        # qg = F.interpolate(qg, size=(H1p, W1p), mode='bilinear', align_corners=False)
        q_s = q_s + qg
        q_s = q_s + self.mlp(q_s)
        q_s = F.interpolate(q_s, size=(H, W), mode='bilinear', align_corners=False)
        out = shortcut + self.out_conv(q_s)
        return out

if __name__ == '__main__':
    import torch
    # model = ResnetGenerator_timestep_restime_text()
    # print(model)
    # print(model.init_conv)

    # x = torch.randn(2, 3, 64, 64)
    # time = torch.tensor([2, 5])
    # output = model(x, time)
    # print(output.shape)
    # attn = LinearAttention(64, heads = 4, dim_head = 32, num_mem_kv = 4)
    attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
    x = torch.randn(2, 64, 64, 64)
    B, C, H, W = x.shape
    x = x.view(B, C, H*W).permute(0, 2, 1)
    out = attn(x, x, x)[0]
    out = out.permute(1, 2, 0).contiguous().view(B, C, H, W)
    print(attn, out.shape)
