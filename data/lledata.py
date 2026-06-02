import torch
import numpy as np
import os
from PIL import Image
import random


class LLEData(torch.utils.data.Dataset):
    def __init__(self, opt, inp_path, gt_path=None):
        super(LLEData, self).__init__()
        # 只加载图像文件，过滤掉 .DS_Store 等非图像文件
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        self.img_li = [path for path in os.listdir(inp_path) 
                       if os.path.splitext(path.lower())[1] in img_extensions]
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.opt = opt
        # 由于采用仿射变换架构，可以直接输入较大分辨率图像进行端到端训练
        # 设置patch_size为0表示不进行随机裁剪，保留完整图像
        self.patch_size = opt.config['train'].get('patch_size', 0) if hasattr(opt, 'config') else 0
        # Gamma 校正范围
        self.gamma_range = opt.config['train'].get('gamma_range', [0.7, 1.3]) if hasattr(opt, 'config') else [0.7, 1.3]
        self.use_gamma = opt.config['train'].get('use_gamma', True) if hasattr(opt, 'config') else True
        # 多尺度裁剪配置
        self.global_crop_prob = opt.config['train'].get('global_crop_prob', 0.3) if hasattr(opt, 'config') else 0.3
        self.scale_range = opt.config['train'].get('scale_range', [0.2, 0.5]) if hasattr(opt, 'config') else [0.2, 0.5]

    def random_crop(self, inp, gt):
        """随机裁剪相同位置的patch"""
        h, w = inp.shape[1], inp.shape[2]
        ps = self.patch_size
        
        # 如果图片小于patch_size，直接返回原图
        if h <= ps or w <= ps:
            return inp, gt
        
        # 随机选择裁剪位置
        hh = random.randint(0, h - ps)
        ww = random.randint(0, w - ps)
        
        inp = inp[:, hh:hh+ps, ww:ww+ps]
        gt = gt[:, hh:hh+ps, ww:ww+ps]
        
        return inp, gt

    def multi_scale_crop(self, inp, gt, scale_range=(0.2, 0.5)):
        """
        多尺度裁剪：先缩放再裁剪
        
        Args:
            inp: 输入图像 [C, H, W]
            gt: GT图像 [C, H, W]
            scale_range: 缩放比例范围 (min_scale, max_scale)
        
        Returns:
            inp_cropped: 裁剪后的输入图像 [C, ps, ps]
            gt_cropped: 裁剪后的GT图像 [C, ps, ps]
        """
        ps = self.patch_size
        h, w = inp.shape[1], inp.shape[2]
        
        # 随机选择缩放比例
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # 计算缩放后的尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 确保缩放后的尺寸至少为 patch_size
        if new_h < ps or new_w < ps:
            # 调整缩放比例，确保缩放后至少有一边 >= ps
            scale = max(ps / h, ps / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        
        # 使用双线性插值缩放 Input 和 GT（空间变换绑定）
        inp_scaled = torch.nn.functional.interpolate(
            inp.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        gt_scaled = torch.nn.functional.interpolate(
            gt.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 在缩放后的图上进行随机裁剪
        if new_h >= ps and new_w >= ps:
            # 随机选择裁剪位置
            hh = random.randint(0, new_h - ps)
            ww = random.randint(0, new_w - ps)
            
            inp_cropped = inp_scaled[:, hh:hh+ps, ww:ww+ps]
            gt_cropped = gt_scaled[:, hh:hh+ps, ww:ww+ps]
        else:
            # 如果缩放后某一边小于 ps，使用 Pad 补齐
            pad_h = max(0, ps - new_h)
            pad_w = max(0, ps - new_w)
            
            inp_padded = torch.nn.functional.pad(inp_scaled, (0, pad_w, 0, pad_h), mode='reflect')
            gt_padded = torch.nn.functional.pad(gt_scaled, (0, pad_w, 0, pad_h), mode='reflect')
            
            # 随机选择裁剪位置
            hh = random.randint(0, inp_padded.shape[1] - ps)
            ww = random.randint(0, inp_padded.shape[2] - ps)
            
            inp_cropped = inp_padded[:, hh:hh+ps, ww:ww+ps]
            gt_cropped = gt_padded[:, hh:hh+ps, ww:ww+ps]
        
        return inp_cropped, gt_cropped

    def gamma_correction(self, inp):
        """随机 Gamma 校正数据增强"""
        if not self.use_gamma:
            return inp
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        # I_new = I^gamma
        inp = torch.pow(inp.clamp(1e-8, 1.0), gamma)
        return inp

    def random_flip(self, inp, gt):
        """随机翻转数据增强"""
        # 水平翻转
        if random.random() > 0.5:
            inp = torch.flip(inp, dims=[2])
            gt = torch.flip(gt, dims=[2])
        # 垂直翻转
        if random.random() > 0.5:
            inp = torch.flip(inp, dims=[1])
            gt = torch.flip(gt, dims=[1])
        return inp, gt

    def __getitem__(self, index):
        inp = Image.open(os.path.join(self.inp_path, self.img_li[index]))
        inp = np.array(inp).transpose([2, 0, 1])
        inp = inp.astype(np.float32) / 255

        inp = torch.from_numpy(inp)  # 使用 from_numpy，数据在 CPU 上

        if self.gt_path: # gt_path -> train/test not demo
            gt = Image.open(os.path.join(self.gt_path, self.img_li[index]))
            gt = np.array(gt).transpose([2, 0, 1])
            gt = gt.astype(np.float32) / 255

            gt = torch.from_numpy(gt)  # 使用 from_numpy，数据在 CPU 上
            
            # 训练时随机裁剪
            if self.patch_size > 0:
                # 多尺度混合裁剪：30% 概率全局视野，70% 概率局部细节
                if random.random() < self.global_crop_prob:
                    # 全局视野裁剪：先缩放再裁剪
                    inp, gt = self.multi_scale_crop(inp, gt, scale_range=tuple(self.scale_range))
                else:
                    # 局部细节裁剪：直接在原图上裁剪
                    inp, gt = self.random_crop(inp, gt)
            
            # 随机翻转
            inp, gt = self.random_flip(inp, gt)
            
            # Gamma 校正（只对输入，不对GT）
            inp = self.gamma_correction(inp)

            return inp, gt, self.img_li[index].split('.')[0]
        return inp, self.img_li[index].split('.')[0]

    def __len__(self):
        return len(self.img_li)
