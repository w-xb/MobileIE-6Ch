import torch.nn as nn
import torch
from .utils_IWO import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
)

class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )
        self.att = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1= nn.Sequential( 
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail = MBRConv3(channels, 6, rep_scale=rep_scale)  # 6通道: 3光照 + 3去噪残差
        self.tail_warm = MBRConv3(channels, 6, rep_scale=rep_scale)
        self.drop = DropBlock(3)
        
    def forward(self, x):
        x_in = x  # 保存原始输入
        x0 = self.head(x)
        x1 = self.body(x0)      
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1 , dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        # 带残差去噪的 Retinex
        out = self.tail(x4)
        illu_map = torch.sigmoid(out[:, 0:3, :, :]) * 0.99 + 0.01  # 前3通道: RGB光照图
        noise_res = out[:, 3:6, :, :]  # 后3通道: 去噪残差
        # 除法恢复亮度 + 残差抵消放大的噪声
        enhanced = (x_in / illu_map) + noise_res
        return enhanced

    def forward_warm(self, x):
        x_in = x  # 保存原始输入
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        # 带残差去噪的 Retinex
        out1 = self.tail(x)
        illu1 = torch.sigmoid(out1[:, 0:3, :, :]) * 0.99 + 0.01  # 前3通道: RGB光照图
        res1 = out1[:, 3:6, :, :]  # 后3通道: 去噪残差
        enhanced1 = (x_in / illu1) + res1
        
        out2 = self.tail_warm(x)
        illu2 = torch.sigmoid(out2[:, 0:3, :, :]) * 0.99 + 0.01  # 前3通道: RGB光照图
        res2 = out2[:, 3:6, :, :]  # 后3通道: 去噪残差
        enhanced2 = (x_in / illu2) + res2
        return enhanced1, enhanced2

    def slim(self):
        net_slim = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
               if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w 
                    weight_slim['%s.bias' % name] = b 
            elif isinstance(mod, FST):
                weight_slim['%s.bias' % name] = mod.bias
                weight_slim['%s.weight1' % name] = mod.weight1
                weight_slim['%s.weight2' % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim

class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential( 
            nn.Conv2d(1, channels, 1, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(channels, 6, 3, 1, 1)  # 6通道: 3光照 + 3去噪残差
        
    def forward(self, x):
        x_in = x  # 保存原始输入
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        # 带残差去噪的 Retinex
        out = self.tail(x4)
        illu_map = torch.sigmoid(out[:, 0:3, :, :]) * 0.99 + 0.01  # 前3通道: RGB光照图
        noise_res = out[:, 3:6, :, :]  # 后3通道: 去噪残差
        enhanced = (x_in / illu_map) + noise_res
        return enhanced

    def forward_warm(self, x):
        x_in = x  # 保存原始输入
        x = self.head(x)
        x = self.body(x)
        # 带残差去噪的 Retinex
        out = self.tail(x)
        illu_map = torch.sigmoid(out[:, 0:3, :, :]) * 0.99 + 0.01  # 前3通道: RGB光照图
        noise_res = out[:, 3:6, :, :]  # 后3通道: 去噪残差
        enhanced = (x_in / illu_map) + noise_res
        return enhanced, enhanced

    def slim(self):
        """Slim version already, return self"""
        return self
