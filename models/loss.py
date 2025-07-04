import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda, epoch=None, warp_out=None):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())    
        if len(output['likelihoods']) > 2:
            out["bpp_joint"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][2].values())  
            out['bpp0'] = out['bpp0'] + out["bpp_joint"]/2
            out["bpp1"] = out["bpp1"] + out["bpp_joint"]/2   


        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = lmbda * (out["mse0"] + out["mse1"])/2        #end to end
        out['loss'] = out['mse_loss'] + out['bpp_loss']

        return out



class MS_SSIM_Loss(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda, epoch=None, warp_out=None):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())    
        if len(output['likelihoods']) > 2:
            out["bpp_joint"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][2].values())  
            out['bpp0'] = out['bpp0'] + out["bpp_joint"]/2
            out["bpp1"] = out["bpp1"] + out["bpp_joint"]/2   

        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2

        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1- self.ms_ssim(output['x_hat'][1], target2)
 
        out['ms_ssim_loss'] = (out["ms_ssim0"] + out["ms_ssim1"])/2        #end to end
        out['loss'] = lmbda * out['ms_ssim_loss'] + out['bpp_loss']
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, max_val=255, device_id=0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device_id = device_id

    def _ssim(self, img1, img2):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        if self.device_id != None:
            window = window.cuda(self.device_id)

        mu1 = F.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        msssim=Variable(torch.Tensor(levels,))
        mcs=Variable(torch.Tensor(levels,))
        # if self.device_id != None:
        #     weight = weight.cuda(self.device_id)
        #     weight = msssim.cuda(self.device_id)
        #     weight = mcs.cuda(self.device_id)
        #     print(weight.device)

        for i in range(levels):
            ssim_map, mcs_map=self._ssim(img1, img2)
            msssim[i]=ssim_map
            mcs[i]=mcs_map
            filtered_im1=F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2=F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1=filtered_im1
            img2=filtered_im2

        value=(torch.prod(mcs[0:levels-1]**weight[0:levels-1]) *
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2, levels=5):
        return self.ms_ssim(img1, img2, levels)

