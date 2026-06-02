import torch
import torch.nn as nn
import torch.nn.functional as F
from option import get_option

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, inp, target):
        return torch.sqrt((inp - target) ** 2 + self.eps2).mean()


class SSIMLoss(nn.Module):
    """SSIM Loss: 1 - SSIM(out, gt)"""
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
    
    def forward(self, out, gt):
        mu1 = F.avg_pool2d(out, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(gt, self.window_size, stride=1, padding=self.window_size//2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(out * out, self.window_size, stride=1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(gt * gt, self.window_size, stride=1, padding=self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(out * gt, self.window_size, stride=1, padding=self.window_size//2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return 1 - ssim_map.mean()


class ColorLoss(nn.Module):
    """余弦相似度损失：将 out 和 gt 展平为 (B, 3, H*W)，计算 cosine_similarity"""
    def __init__(self):
        super(ColorLoss, self).__init__()
    
    def forward(self, out, gt):
        B, C, H, W = out.shape
        out_flat = out.view(B, C, -1)  # (B, 3, H*W)
        gt_flat = gt.view(B, C, -1)
        cosine = F.cosine_similarity(out_flat, gt_flat, dim=2)  # (B, H*W)
        return (1 - cosine).mean()


class FrequencyLoss(nn.Module):
    """傅里叶频率损失：对 out 和 gt 做 fft2，取幅度谱 abs()，计算 L1 Loss"""
    def __init__(self):
        super(FrequencyLoss, self).__init__()
    
    def forward(self, out, gt):
        out_fft = torch.fft.fft2(out, norm='ortho')
        gt_fft = torch.fft.fft2(gt, norm='ortho')
        out_mag = torch.abs(out_fft)
        gt_mag = torch.abs(gt_fft)
        return F.l1_loss(out_mag, gt_mag)
#####################################################################################################
class OutlierAwareLoss(nn.Module):
    def __init__(self,):
        super(OutlierAwareLoss, self).__init__()

    def forward(self, out, lab):
        delta = out - lab
        var = delta.std((2, 3), keepdims=True) / (2 ** .5)
        avg = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()       
        loss = (delta.abs() * weight).mean()
        return loss
    
#####################################################################################################
class LossWarmup(nn.Module):
    def __init__(self):
        super(LossWarmup, self).__init__()
        self.loss_cb = CharbonnierLoss(1e-8)
        self.loss_cs = nn.CosineSimilarity()    

    def forward(self, inp, gt, warmup1, warmup2):
        loss = self.loss_cb(warmup2, inp) + \
               (self.loss_cb(warmup1, gt) + (1 - self.loss_cs(warmup1.clip(0, 1), gt)).mean())
        
        return loss 


class LossLLE(nn.Module):
    """
    综合损失函数：
    L_total = λ1*L_char + λ2*L_ssim + λ3*L_color + λ4*L_freq
    """
    def __init__(self, lambda_char=1.0, lambda_ssim=0.5, lambda_color=0.1, lambda_freq=0.1):
        super(LossLLE, self).__init__()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_freq = lambda_freq
        
        self.loss_char = CharbonnierLoss(eps=1e-3)
        self.loss_ssim = SSIMLoss()
        self.loss_color = ColorLoss()
        self.loss_freq = FrequencyLoss()
    
    def forward(self, out, gt):
        out_clamped = torch.clamp(out, 0.0, 1.0)
        gt_clamped = torch.clamp(gt, 0.0, 1.0)
        
        loss_char = self.loss_char(out_clamped, gt_clamped)
        loss_ssim = self.loss_ssim(out_clamped, gt_clamped)
        loss_color = self.loss_color(out_clamped, gt_clamped)
        loss_freq = self.loss_freq(out_clamped, gt_clamped)
        
        total = self.lambda_char * loss_char + \
                self.lambda_ssim * loss_ssim + \
                self.lambda_color * loss_color + \
                self.lambda_freq * loss_freq
        return total
        
class LossISP(nn.Module):
    def __init__(self):
        super(LossISP, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean()) + 2 * self.psnr(out, gt) 
        return loss

def import_loss(training_task):
    if training_task == 'isp':
        return LossISP()
    elif training_task == 'lle':
        return LossLLE()
    elif training_task == 'warmup':
        return LossWarmup()
    else:
        raise ValueError('unknown training task, please choose from [isp, lle, warmup].')

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        imdff=pred-target
        rmse=((imdff**2).mean(dim=(1,2,3))+1e-8).sqrt()
        loss=20*torch.log10(1/rmse).mean()
        loss=(50.0-loss)/100.0
        return loss    
