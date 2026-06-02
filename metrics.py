"""
图像质量评估指标模块
包含：SSIM, LPIPS, DISTS (全参考) 和 LIQE, MUSIQ, Q-Align (无参考)
"""
import torch
import torch.nn.functional as F
import numpy as np


class SSIMCalculator:
    """SSIM 计算器"""
    def __init__(self, window_size=11):
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def __call__(self, out, gt):
        """返回 SSIM 值 (越高越好, 范围 0-1)"""
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
        return ssim_map.mean().item()


class LPIPSCalculator:
    """LPIPS 计算器 (需要 lpips 库)"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import lpips
                self.model = lpips.LPIPS(net='alex').to(self.device)
                self.model.eval()
            except ImportError:
                print("Warning: lpips not installed. Run: pip install lpips")
                return None
        return self.model
    
    def __call__(self, out, gt):
        """返回 LPIPS 值 (越低越好, 范围 0-1)"""
        model = self._load_model()
        if model is None:
            return 0.5  # 默认值
        with torch.no_grad():
            # LPIPS 需要 [-1, 1] 范围
            out_norm = out * 2 - 1
            gt_norm = gt * 2 - 1
            lpips_val = model(out_norm, gt_norm)
            return lpips_val.mean().item()


class LIQECalculator:
    """LIQE 无参考图像质量评估 (需要 pyiqa 库)"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import pyiqa
                self.model = pyiqa.create_metric('liqe', device=self.device)
            except ImportError:
                print("Warning: pyiqa not installed. Run: pip install pyiqa")
                return None
        return self.model
    
    def __call__(self, img):
        """返回 LIQE 值 (越高越好, 范围约 0-100)"""
        model = self._load_model()
        if model is None:
            return 50.0  # 默认值
        with torch.no_grad():
            score = model(img)
            return score.mean().item()


class MUSIQCalculator:
    """MUSIQ 无参考图像质量评估"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import pyiqa
                self.model = pyiqa.create_metric('musiq', device=self.device)
            except ImportError:
                print("Warning: pyiqa not installed. Run: pip install pyiqa")
                return None
        return self.model
    
    def __call__(self, img):
        """返回 MUSIQ 值 (越高越好)"""
        model = self._load_model()
        if model is None:
            return 50.0
        with torch.no_grad():
            score = model(img)
            return score.mean().item()


class DISTSCalculator:
    """DISTS 全参考图像质量评估"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import pyiqa
                self.model = pyiqa.create_metric('dists', device=self.device)
            except ImportError:
                print("Warning: pyiqa not installed. Run: pip install pyiqa")
                return None
        return self.model
    
    def __call__(self, out, gt):
        """返回 DISTS 值 (越低越好, 范围 0-1)"""
        model = self._load_model()
        if model is None:
            return 0.5
        with torch.no_grad():
            score = model(out, gt)
            return score.mean().item()


class QAlignCalculator:
    """Q-Align 无参考图像质量评估 (较慢)"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import pyiqa
                self.model = pyiqa.create_metric('qalign', device=self.device)
            except ImportError:
                print("Warning: pyiqa not installed or qalign not available")
                return None
        return self.model
    
    def __call__(self, img):
        """返回 Q-Align 值 (越高越好)"""
        model = self._load_model()
        if model is None:
            return 50.0
        with torch.no_grad():
            score = model(img)
            return score.mean().item()


class FastMetrics:
    """
    阶段1快速指标计算器 (训练时使用)
    计算: PSNR, SSIM
    Score_fast = SSIM
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.ssim = SSIMCalculator()
    
    def compute(self, out, gt):
        """计算快速评估指标"""
        ssim_val = self.ssim(out, gt)
        
        # 计算 PSNR
        mse = ((out - gt) ** 2).mean()
        psnr_val = 10 * torch.log10(1.0 / (mse + 1e-8))
        
        # Score_fast = SSIM
        score_fast = ssim_val
        
        return {
            'ssim': ssim_val,
            'psnr': psnr_val.item(),
            'lpips': 0.0,
            'liqe': 0.0,
            'score_fast': score_fast
        }


class FullMetrics:
    """
    阶段2完整指标计算器 (训练后使用)
    计算全部6个指标: SSIM, LPIPS, DISTS, LIQE, MUSIQ, Q-Align
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.ssim = SSIMCalculator()
        self.lpips = LPIPSCalculator(device)
        self.dists = DISTSCalculator(device)
        self.liqe = LIQECalculator(device)
        self.musiq = MUSIQCalculator(device)
        self.qalign = QAlignCalculator(device)
    
    def compute(self, out, gt):
        """计算完整评估指标"""
        ssim_val = self.ssim(out, gt)
        lpips_val = self.lpips(out, gt)
        dists_val = self.dists(out, gt)
        liqe_val = self.liqe(out)
        musiq_val = self.musiq(out)
        qalign_val = self.qalign(out)
        
        # 归一化评分
        # SSIM: 0-1, 越高越好
        # LPIPS: 0-1, 越低越好 -> 1-LPIPS
        # DISTS: 0-1, 越低越好 -> 1-DISTS
        # LIQE: 0-100, 越高越好 -> /100
        # MUSIQ: 0-100, 越高越好 -> /100
        # Q-Align: 0-5, 越高越好 -> /5
        
        norm_ssim = ssim_val
        norm_lpips = 1 - lpips_val
        norm_dists = 1 - dists_val
        norm_liqe = liqe_val / 100.0
        norm_musiq = musiq_val / 100.0
        norm_qalign = qalign_val / 5.0
        
        score_full = norm_ssim + norm_lpips + norm_dists + norm_liqe + norm_musiq + norm_qalign
        
        return {
            'ssim': ssim_val,
            'lpips': lpips_val,
            'dists': dists_val,
            'liqe': liqe_val,
            'musiq': musiq_val,
            'qalign': qalign_val,
            'score_full': score_full,
            'norm': {
                'ssim': norm_ssim,
                'lpips': norm_lpips,
                'dists': norm_dists,
                'liqe': norm_liqe,
                'musiq': norm_musiq,
                'qalign': norm_qalign
            }
        }


class TopKModelTracker:
    """跟踪 Top-K 最佳模型"""
    def __init__(self, k=3):
        self.k = k
        self.score_models = []  # (score, epoch, metrics) - Score_fast
        self.ssim_models = []   # (ssim, epoch, metrics) - SSIM
    
    def update(self, epoch, metrics):
        """
        更新 Top-K 列表
        返回: (should_save_score, should_save_ssim)
        """
        score = metrics['score_fast']
        ssim = metrics['ssim']
        
        should_save_score = False
        should_save_ssim = False
        
        # 检查是否进入 Score Top-K
        if len(self.score_models) < self.k:
            self.score_models.append((score, epoch, metrics))
            self.score_models.sort(key=lambda x: x[0], reverse=True)
            should_save_score = True
        elif score > self.score_models[-1][0]:
            self.score_models[-1] = (score, epoch, metrics)
            self.score_models.sort(key=lambda x: x[0], reverse=True)
            should_save_score = True
        
        # 检查是否进入 SSIM Top-K
        if len(self.ssim_models) < self.k:
            self.ssim_models.append((ssim, epoch, metrics))
            self.ssim_models.sort(key=lambda x: x[0], reverse=True)
            should_save_ssim = True
        elif ssim > self.ssim_models[-1][0]:
            self.ssim_models[-1] = (ssim, epoch, metrics)
            self.ssim_models.sort(key=lambda x: x[0], reverse=True)
            should_save_ssim = True
        
        return should_save_score, should_save_ssim
    
    def get_rank(self, epoch, by='score'):
        """获取指定 epoch 在 Top-K 中的排名 (1-based), 不在则返回 -1"""
        models = self.score_models if by == 'score' else self.ssim_models
        for i, (_, ep, _) in enumerate(models):
            if ep == epoch:
                return i + 1
        return -1
