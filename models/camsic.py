import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from compressai.ops import quantize_ste as ste_round
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.models import CompressionModel, get_scale_table
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
import math
from torch.autograd import Variable, Function
from compressai.layers import GDN, MaskedConv2d
from compressai.zoo import *
from .elic import get_elic_models, ELIC


class MaskEntropyModel(nn.Module):
    def __init__(self, num_channels=192, context_len=1, embed_dim=768, depths=[4], num_heads=12, 
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
        attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, denominator=8):
        super().__init__()

        self.window_size = window_size
        self.hyper_model = ConvHyperPrior(y_dim=num_channels, z_dim=num_channels//2)

        self.inital_frame = nn.Parameter(torch.zeros(1, num_channels, 1, 1).uniform_(-3,3))
        self.project = nn.Linear(num_channels, embed_dim)
        self.post_project_norm = norm_layer(embed_dim)

        self.temporal_position = nn.Parameter(torch.zeros(context_len+1, embed_dim))
        trunc_normal_(self.temporal_position, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layer = BasicLayer(dim=[embed_dim], depth=depths[0], num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, context_dim=0, context_heads=0, 
            context=False, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
            attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
            downsample=None, inverse=False, use_rpe=True)   

        self.layer._init_respostnorm()
 
        self.entropy_parameters = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, num_channels*2),
        )
        self.fusion = FeaEncoder(in_ch=num_channels*2, out_ch=num_channels, kernel_size=3, stride=1)
        self.gaussian_conditional = GaussianConditional(None)
        self.gamma = self.gamma_func("sine")
        self.denominator = denominator

    def forward(self, x, x_ref=None):
        if self.training:
            x_hat, likelihoods = self.forward_train(x, x_ref)
        else:
            x_hat, likelihoods = self.forward_test(x, x_ref)
        return x_hat, likelihoods

    def forward_train(self, x, x_ref=None):
        #x:[B, y_dim, H, W], joint_params: [B, 3*H, W, middle_dim]
        B, C, H, W = x.size()
        if x_ref is None:
            x_ref = self.inital_frame.expand(B, -1, H, W)

        hyper_params, z_likelihoods, _ = self.hyper_model(x)
        x_fusion = self.fusion(torch.cat([hyper_params, x_ref], dim=1))

        mask_gaussian_params = self.forward_prior(x_fusion, None)
        mask_scales_hat, mask_means_hat = mask_gaussian_params.chunk(2, 1)
        mask = torch.randint(low=0, high=2, size=(1, 1, H, W), dtype=torch.bool, device=x.device) #spatial
        mask_scales_hat = mask_scales_hat * mask
        mask_means_hat = mask_means_hat * mask
        mix_x = ste_round(x * mask - mask_means_hat) + mask_means_hat + x_fusion * (~mask)
        
        visible_gaussian_params = self.forward_prior(mix_x, mask)
        scales_hat, means_hat = visible_gaussian_params.chunk(2, 1)
        scales_hat = scales_hat * (~mask) + mask_scales_hat
        means_hat = means_hat * (~mask) + mask_means_hat
        _, y_likelihoods = self.gaussian_conditional(x, scales_hat, means=means_hat)
        x_hat = ste_round(x - means_hat) + means_hat
        return x_hat, {"y": y_likelihoods, "z":z_likelihoods}

    def forward_test(self, x, x_ref=None):
        B, C, H, W = x.size()
        if x_ref is None:
            x_ref = self.inital_frame.expand(B, -1, H, W)
        
        hyper_params, z_likelihoods, _ = self.hyper_model(x)
        x_fusion = self.fusion(torch.cat([hyper_params, x_ref], dim=1))
        mask_x = x_fusion
        real_scales_hat, real_means_hat = 0, 0
        accumulate_mask = None
        for i in range(1, self.denominator+1):
            if i == self.denominator:
                decode_number = int(torch.sum(~accumulate_mask))
            else:
                decode_number = math.floor((self.gamma(i/self.denominator)-self.gamma((i-1)/self.denominator)) * H * W)

            scales_hat, means_hat, mask, accumulate_mask = self.forward_regressive_mask(mask_x, 
                    decode_number, pre_accumulate_mask=accumulate_mask)

            mask_x = ste_round(x * mask - means_hat) + means_hat + mask_x * (~mask) 
            real_scales_hat = real_scales_hat + scales_hat
            real_means_hat = real_means_hat + means_hat

        _, y_likelihoods = self.gaussian_conditional(x, real_scales_hat, means=real_means_hat)
        x_hat = ste_round(x - real_means_hat) + real_means_hat
        return x_hat, {"y": y_likelihoods, "z":z_likelihoods}

    def aux_loss(self):
        return self.hyper_model.aux_loss()

    def forward_prior(self, x, mask=None):
        B, _, H, W = x.size()
        x = x.view(B, -1, H*W).transpose(1, 2)
        x = self.project(x)
        x = self.post_project_norm(x)
        B, _, C = x.size()
        #x = x + self.position
        if mask is None:
            x = x + self.temporal_position[0].unsqueeze(0).unsqueeze(0)
        else:
            mask = mask.view(1, 1, -1).transpose(1, 2).contiguous().expand(1, -1, C)
            temporal_position = (~mask)* (self.temporal_position[0].unsqueeze(0).unsqueeze(0)) + mask * (self.temporal_position[1].unsqueeze(0).unsqueeze(0))
            x = x + temporal_position

        params, H, W = self.layer(x, H, W) #[B, H*W, C]
        gaussian_params = self.entropy_parameters(params)
        gaussian_params = gaussian_params.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[B, C*2, H, W]
        return gaussian_params

    def gamma_func(self, mode="sine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "sine":
            return lambda r: np.sin(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError
       
    def forward_regressive_mask(self, x, decode_number, pre_accumulate_mask=None):
        B, _, H, W = x.size()
        gaussian_params = self.forward_prior(x, pre_accumulate_mask)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        bits = self.get_bits(means_hat, scales_hat)
        bits = torch.sum(bits, dim=1)
        if pre_accumulate_mask is not None:
            bits = bits.masked_fill(pre_accumulate_mask == True, 1e9)
        bits = -bits.view(B, -1)
        sample = bits.topk(decode_number, dim=-1).indices
        mask = torch.zeros((1, H*W), dtype=torch.bool, device=x.device)
        mask.scatter_(dim=1, index=sample, value=True)
        mask = mask.view(1, 1, H, W)

        if pre_accumulate_mask is None:
            accumulate_mask = mask
        else:      
            accumulate_mask = mask + pre_accumulate_mask
        #print(torch.max(accumulate_mask))
        scales_hat = scales_hat * mask
        means_hat = means_hat * mask
        return scales_hat, means_hat, mask, accumulate_mask

    def get_bits(self, mu, sigma):
        sigma = self.gaussian_conditional.lower_bound_scale(sigma)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(mu + 0.5) - gaussian.cdf(mu - 0.5)
        probs = self.gaussian_conditional.likelihood_lower_bound(probs)
        bits = -1.0 * torch.log(probs) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    def compress(self, x, x_ref=None):
        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        B, C, H, W = x.size()
        if x_ref is None:
            x_ref = self.inital_frame.expand(B, -1, H, W)

        hyper_params, z_info_dict = self.hyper_model.compress(x)
        x_fusion = self.fusion(torch.cat([hyper_params, x_ref], dim=1))
        mask_x = x_fusion
        real_scales_hat, real_means_hat = 0, 0
        x_hat = 0
        accumulate_mask = None
        for i in range(1, self.denominator+1):
            if i == self.denominator:
                decode_number = int(torch.sum(~accumulate_mask))
            else:
                decode_number = math.floor((self.gamma(i/self.denominator)-self.gamma((i-1)/self.denominator)) * H * W)

            scales_hat, means_hat, mask, accumulate_mask = self.forward_regressive_mask(mask_x, decode_number, pre_accumulate_mask=accumulate_mask)
            mask_hat = self.compress_squeeze(x, mask, scales_hat, means_hat, symbols_list, indexes_list)
            mask_x = mask_hat + mask_x * (~mask)
            x_hat = x_hat + mask_hat

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        return x_hat, {"strings": [y_strings, z_info_dict["strings"]], "shape": z_info_dict["shape"]}

    def decompress(self, strings, shape, x_ref=None):
        assert isinstance(strings, list) and len(strings) == 2
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()        
        y_strings = strings[0][0]
        z_strings = strings[1]
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        hyper_params = self.hyper_model.decompress(z_strings, shape)
        B, _, H, W = hyper_params.size()
        if x_ref is None:
            x_ref = self.inital_frame.expand(B, -1, H, W)

        x_fusion = self.fusion(torch.cat([hyper_params, x_ref], dim=1))
        mask_x = x_fusion
        real_scales_hat, real_means_hat = 0, 0
        x_hat = 0
        accumulate_mask = None
        for i in range(1, self.denominator+1):
            if i == self.denominator:
                decode_number = int(torch.sum(~accumulate_mask))
            else:
                decode_number = math.floor((self.gamma(i/self.denominator)-self.gamma((i-1)/self.denominator)) * H * W)

            scales_hat, means_hat, mask, accumulate_mask = self.forward_regressive_mask(mask_x, decode_number, pre_accumulate_mask=accumulate_mask)
            mask_hat = self.decompress_squeeze(mask, scales_hat, means_hat, decoder, cdf, cdf_lengths, offsets)
            mask_x = mask_hat + mask_x * (~mask)
            x_hat = x_hat + mask_hat

        torch.cuda.synchronize()
        return x_hat

    def mim_sequeeze(self, y, mask):
        B, C, _, _ = y.shape
        mask = mask.view(-1)
        nonzero_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        #y_squeeze = torch.index_select(y, dim=-1, index=nonzero_idx)
        y_squeeze = y.view(B, C, -1)[:,:, nonzero_idx]
        return y_squeeze

    def mim_unsequeeze(self, y_squeeze, mask):
        B, C, _ = y_squeeze.shape
        H, W = mask.size()[-2:]
        y = torch.zeros([B, C, H*W], dtype=y_squeeze.dtype).to(y_squeeze.device)
        nonzero_idx = torch.nonzero(mask.view(-1), as_tuple=False).view(-1)
        y[:,:,nonzero_idx] = y_squeeze
        y = y.view(B, C, H, W).float()
        return y

    def compress_squeeze(self, y, mask, scales_hat, means_hat, symbols_list, indexes_list):
        B, C, _, _ = y.shape
        H, W = mask.size()[-2:]
        mask = mask.view(-1)
        nonzero_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        y_squeeze = y.view(B, C, -1)[:,:, nonzero_idx]
        scales_squeeze = scales_hat.view(B, C, -1)[:,:, nonzero_idx]
        means_squeeze = means_hat.view(B, C, -1)[:,:, nonzero_idx]
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        y_squeeze_hat = self.gaussian_conditional.quantize(y_squeeze, "symbols", means_squeeze)
        symbols_list.extend(y_squeeze_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())

        y_squeeze = y_squeeze_hat + means_squeeze
        y_hat = torch.zeros([B, C, H*W], dtype=y_squeeze.dtype).to(y_squeeze.device)
        y_hat[:,:,nonzero_idx] = y_squeeze
        y_hat = y_hat.view(B, C, H, W).float()
        #y_hat = self.mim_unsequeeze(y_squeeze_hat + means_squeeze, mask) 
        return y_hat

    def decompress_squeeze(self, mask, scales_hat, means_hat, decoder, cdf, cdf_lengths, offsets):
        B, C, _, _ = means_hat.shape
        H, W = mask.size()[-2:]
        mask = mask.view(-1)
        nonzero_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        scales_squeeze = scales_hat.view(B, C, -1)[:,:, nonzero_idx]
        means_squeeze = means_hat.view(B, C, -1)[:,:, nonzero_idx]
        #scales_squeeze = self.mim_sequeeze(scales_hat, mask)
        #means_squeeze = self.mim_sequeeze(means_hat, mask)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        y_squeeze = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        y_squeeze = torch.Tensor(y_squeeze).reshape(scales_squeeze.shape).to(scales_hat.device) + means_squeeze
        #y_hat = self.mim_unsequeeze(y_hat, mask)
        
        y_hat = torch.zeros([B, C, H*W], dtype=y_squeeze.dtype).to(y_squeeze.device)
        y_hat[:,:,nonzero_idx] = y_squeeze
        y_hat = y_hat.view(B, C, H, W).float()
        return y_hat


def crop(x, padding):
    return F.pad(x, tuple(-p for p in padding))


def window_split(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size) #, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_merge(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


class CAMSIC(nn.Module):
    def __init__(self, arch_name = "ELIC", num_channels=320, context_len=1, embed_dim=768, depths=[4], num_heads=12, 
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
        attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, scratch=True, denominator=8):
        super().__init__()

        if arch_name == "ELIC":
            self.g_a, self.g_s = get_elic_models(192, 320, 3)
        elif arch_name == "mbt2018":
            self.g_a, self.g_s = get_mbt_models(192, num_channels)
        elif arch_name == "cheng2020-anchor":
            self.g_a, self.g_s = get_cheng_models(num_channels)

        self.arch_name = arch_name
        self.entropy_model = MaskEntropyModel(num_channels=num_channels, context_len=context_len, embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, denominator=denominator)
        self.scratch = scratch   
        self.init_weights()

    def forward(self, frames):
        if not isinstance(frames, list):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")
        x_left, x_right = frames
        if "joint" in self.arch_name:
            y_left, y_right = self.g_a(x_left, x_right)
            #print(y_left.shape, y_right.shape)
            y_right_hat, y_right_likelihoods = self.entropy_model(y_right, None)
            y_left_hat, y_left_likelihoods = self.entropy_model(y_left, y_right_hat)
            x_left_hat, x_right_hat = self.g_s(y_left_hat, y_right_hat)
            x_left_hat, x_right_hat = x_left_hat.clamp(0, 1), x_right_hat.clamp(0, 1)
        else:
            # y_left, y_right = self.g_a(x_left), self.g_a(x_right)
            # y_left_hat, y_left_likelihoods = self.entropy_model(y_left, None)
            # y_right_hat, y_right_likelihoods = self.entropy_model(y_right, y_left_hat)
            # x_left_hat, x_right_hat = self.g_s(y_left_hat), self.g_s(y_right_hat)
            y_left, y_right = self.g_a(x_left), self.g_a(x_right)
            y_right_hat, y_right_likelihoods = self.entropy_model(y_right, None)
            y_left_hat, y_left_likelihoods = self.entropy_model(y_left, y_right_hat)
            x_left_hat, x_right_hat = self.g_s(y_left_hat), self.g_s(y_right_hat)
            x_left_hat, x_right_hat = x_left_hat.clamp(0, 1), x_right_hat.clamp(0, 1)


        # if self.scratch:
        #     x_left_hat = torch.sigmoid(x_left_hat)
        #     x_right_hat = torch.sigmoid(x_right_hat)

        reconstructions = [x_left_hat, x_right_hat]
        frames_likelihoods = [y_left_likelihoods, y_right_likelihoods]
        
        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_one_frame(self, x, y_ref=None, scratch=True):
        """Forward function."""
        y = self.g_a(x)
        y_hat, likelihoods = self.entropy_model(y, y_ref)
        x_hat = self.g_s(y_hat)
        if scratch:
            x_hat = torch.sigmoid(x_hat)#scractch
        return x_hat, likelihoods, y_hat

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.entropy_model.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = self.entropy_model.hyper_model.update(force=force)
        return updated

    def load_model_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_model.gaussian_conditional,
            "entropy_model.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def fix_encoder_decoder(self):
        for p in self.g_a.parameters():
            p.requires_grad = False
        for p in self.g_s.parameters():
            p.requires_grad = False

    def aux_loss(self):
        return [self.entropy_model.aux_loss()]



    def compress(self, frames):
        """Forward function."""
        x_left, x_right = frames
        y_left, y_right = self.g_a(x_left), self.g_a(x_right)
        y_right_hat, right_compress_info = self.entropy_model.compress(y_right, None)
        y_left_hat, left_compress_info = self.entropy_model.compress(y_left, y_right_hat)        
        return [right_compress_info, left_compress_info]

    def decompress(self, compress_info):
        """Forward function."""
        right_compress_info, left_compress_info = compress_info
        y_right_hat = self.entropy_model.decompress(right_compress_info["strings"], right_compress_info["shape"], None)
        y_left_hat = self.entropy_model.decompress(left_compress_info["strings"], left_compress_info["shape"], y_right_hat)
        x_left_hat, x_right_hat = self.g_s(y_left_hat), self.g_s(y_right_hat)
        x_left_hat, x_right_hat = x_left_hat.clamp_(0, 1), x_right_hat.clamp_(0, 1)
        reconstructions = [x_left_hat, x_right_hat]
        return {
            "x_hat": reconstructions,
        }


class ConvHyperPrior(CompressionModel):
    def __init__(self, y_dim, z_dim, out_dim=None):
        super().__init__(entropy_bottleneck_channels=z_dim)
        self.h_a = nn.Sequential(
            conv(y_dim, z_dim, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(z_dim, z_dim),
            nn.LeakyReLU(inplace=True),
            conv(z_dim, z_dim),
        )
        output_channels = y_dim if out_dim is None else out_dim
        self.h_s = nn.Sequential(
            deconv(z_dim, z_dim),
            nn.LeakyReLU(inplace=True),
            deconv(z_dim, z_dim),
            nn.LeakyReLU(inplace=True),
            conv(z_dim, output_channels, stride=1, kernel_size=3),
        )

    def forward(self, y):
        # print(y.shape)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        params = self.h_s(z_hat)
        return params, z_likelihoods, z_hat

    def compress(self, y):
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        return params, {"strings": [z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        z_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        params = self.h_s(z_hat)
        return params


class Latent_Residual_Prediction(nn.Module):
    def __init__(self, num_channels, embed_dim):
        """Instantiates dequantizer."""
        super().__init__()
        self.res1 = nn.Sequential(
            nn.Linear(embed_dim, num_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.res2 = FeaEncoder(in_ch=num_channels, out_ch=num_channels, kernel_size=3, stride=1)

    def forward(self, params, H, W):
        B = params.size(0)
        lrp = self.res1(params).view(B, H, W, -1).permute(0, 3, 1, 2)
        lrp = 0.5 * torch.tanh(self.res2(lrp))
        return lrp

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
        attn_drop=0., proj_drop=0., use_rpe=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_rpe = use_rpe

        if self.use_rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, context_dim=64, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.context_dim = context_dim
        head_dim = context_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, context_dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, context_dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(context_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, mask=None):
        B, HW, C1 = x.shape
        B, N, C2 = kv.shape #C2<C1, N<HW
        #print(f"kv:{kv.shape}, context_dim:{self.context_dim}")
        assert C2 == self.context_dim
        q = self.q(x).reshape(B, HW, self.num_heads, C2//self.num_heads).transpose(1, 2) #permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C2//self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        #3, B, num_heads, N, C//num_heads
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(0)
            attn = attn.masked_fill(mask == 0, -1e9)


        attn = attn.softmax(dim=-1)

        if mask is not None:
            # We use the mask again, to be double sure that no masked dimension
            # affects the output.
            attn = attn.masked_fill(mask == 0, 0)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, HW, C2)
        x = self.proj(x) #(B, HW, C1)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
        context_dim=64, context_heads=4, context=False, qkv_bias=True, qk_scale=None, 
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
        inverse=False, use_rpe=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.context = context
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_rpe=use_rpe)
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.context:
            self.cross_attn = Attention(dim, context_dim=context_dim, num_heads=context_heads, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.cross_norm = norm_layer(dim)
           
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, context_x=None):
        #print(x.shape)
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        
        if self.context and context_x is not None:
            x = x + self.drop_path(self.cross_norm(self.cross_attn(x, context_x)))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., 
        context_dim=64, context_heads=4, context=True, qkv_bias=True,
        qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
        downsample=None, inverse=False, use_rpe=True):
        super().__init__()
        self.window_size = window_size
        self.depth = depth

        # build blocks
        if isinstance(window_size, list):
            win_size1, win_size2 = window_size
            self.shift_size = [win_size1//2, win_size2//2]
        else:
            win_size1, win_size2 = window_size, window_size
            self.shift_size = window_size // 2
        blocks = [
            SwinTransformerBlock(dim=dim[0], num_heads=num_heads,
                window_size=win_size1, 
                shift_size=0 if (i % 2 == 0) else win_size1 // 2,
                mlp_ratio=mlp_ratio, context_dim=context_dim, 
                context_heads=context_heads, context=False if (i % 2 == 0) else context, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, inverse=inverse, use_rpe=use_rpe)
            for i in range(depth//2)]
        blocks.extend([
            SwinTransformerBlock(dim=dim[0], num_heads=num_heads,
                window_size=win_size2, 
                shift_size=0 if (i % 2 == 0) else win_size2 // 2,
                mlp_ratio=mlp_ratio, context_dim=context_dim, 
                context_heads=context_heads, context=False if (i % 2 == 0) else context, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, inverse=inverse, use_rpe=use_rpe)
            for i in range(depth//2)])
        self.blocks = nn.ModuleList(blocks)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(in_dim=dim[0], out_dim=dim[1], norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, context_x=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        if isinstance(self.window_size, list):
            attn_mask1 = self.generate_mask(H, W, self.window_size[0], self.shift_size[0], x.device)
            attn_mask2 = self.generate_mask(H, W, self.window_size[1], self.shift_size[1], x.device)
            for i, blk in enumerate(self.blocks):
                blk.H, blk.W = H, W
                if i < self.depth//2:
                    x = blk(x, attn_mask1, context_x)
                else:
                    x = blk(x, attn_mask2, context_x)
        else:
            attn_mask = self.generate_mask(H, W, self.window_size, self.shift_size, x.device)
            for i, blk in enumerate(self.blocks):
                blk.H, blk.W = H, W
                x = blk(x, attn_mask, context_x)
        
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchMerging):
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            elif isinstance(self.downsample, PatchSplit):
                Wh, Ww = H * 2, W * 2
            return x_down, Wh, Ww
        else:
            return x, H, W

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def generate_mask(self, H, W, window_size, shift_size, device):
        Hp = int(np.ceil(H / window_size)) * window_size
        Wp = int(np.ceil(W / window_size)) * window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class FeaEncoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, kernel_size=5, stride=2):
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.residual = nn.Sequential(
            ResidualBlock(out_ch, out_ch, kernel_size=3),
            ResidualBlock(out_ch, out_ch, kernel_size=3),
            ResidualBlock(out_ch, out_ch, kernel_size=3),
        )

    def forward(self, x):
        x = self.conv(x)
        out = self.residual(x)
        out = out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = conv(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_ch, out_ch, kernel_size=kernel_size, stride=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out
