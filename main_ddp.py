import torch
import torch.distributed as dist
import os
import argparse
from tqdm import tqdm
import numpy as np
import copy

from logger import Logger
from option import get_option, load_yaml, save_yaml
from data import import_loader
from loss import import_loss
from model import import_model
from metrics import FastMetrics


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 初始化影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """更新影子权重"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    
    def apply_shadow(self, model):
        """应用影子权重到模型（用于验证/保存）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """恢复原始权重（用于继续训练）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def create_kfold_datasets(dataset_class, opt, inp_path, gt_path, k_folds=5, current_fold=0):
    """创建K折交叉验证的数据集划分"""
    # 获取所有图片列表
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    all_images = [f for f in os.listdir(inp_path) 
                  if os.path.splitext(f.lower())[1] in img_extensions]
    all_images = sorted(all_images)
    
    n_samples = len(all_images)
    fold_size = n_samples // k_folds
    
    # 计算当前折的验证集索引
    valid_start = current_fold * fold_size
    valid_end = valid_start + fold_size if current_fold < k_folds - 1 else n_samples
    
    valid_indices = list(range(valid_start, valid_end))
    train_indices = [i for i in range(n_samples) if i not in valid_indices]
    
    # 创建训练集和验证集
    train_data = dataset_class(opt, inp_path, gt_path)
    valid_data = dataset_class(opt, inp_path, gt_path)
    
    # 子集采样
    train_data.img_li = [train_data.img_li[i] for i in train_indices]
    valid_data.img_li = [valid_data.img_li[i] for i in valid_indices]
    
    return train_data, valid_data, len(train_indices), len(valid_indices)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_ddp():
    """
    初始化分布式训练环境（使用 torchrun）
    
    DDP (DistributedDataParallel) 工作原理：
    1. 每个进程（GPU）运行相同的代码副本
    2. rank: 进程的全局唯一标识符，范围 [0, world_size-1]
    3. local_rank: 进程在当前节点上的GPU编号
    4. world_size: 总进程数（等于使用的GPU数量）
    5. 进程间通过 NCCL 后端进行通信和同步
    
    环境变量（由 torchrun 自动设置）：
    - RANK: 当前进程的全局 rank
    - LOCAL_RANK: 当前进程在当前节点上的本地 rank
    - WORLD_SIZE: 总进程数
    """
    rank = int(os.environ['RANK'])  # 全局进程ID，用于区分不同进程
    local_rank = int(os.environ['LOCAL_RANK'])  # 本地GPU ID，用于设置当前进程使用的GPU
    world_size = int(os.environ['WORLD_SIZE'])  # 总进程数，等于使用的GPU数量
    
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU
    dist.init_process_group("nccl")  # 初始化进程组，使用NCCL后端（GPU通信）
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """
    清理分布式训练环境
    
    训练结束后需要销毁进程组，释放资源
    """
    dist.destroy_process_group()


def save_checkpoint(state, filepath, rank=0):
    """保存检查点（只在 rank 0 上执行）"""
    if rank == 0:
        torch.save(state, filepath)


def load_checkpoint(filepath, net, optimizer=None, scheduler=None, rank=0):
    """加载检查点"""
    if not os.path.exists(filepath):
        return 0, 0  # 如果没有检查点，从 epoch 0 开始
    
    checkpoint = torch.load(filepath, map_location=f'cuda:{rank}')
    net.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0)
    
    return start_epoch, best_psnr


def train_ddp(opt, logger):
    """
    DDP 训练函数（torchrun 版本）
    
    DDP 训练流程：
    1. 初始化分布式环境，获取 rank、local_rank、world_size
    2. 创建数据加载器，使用 DistributedSampler 确保每个进程处理不同的数据
    3. 创建模型并使用 DDP 包装
    4. 训练循环中，每个进程独立计算梯度
    5. DDP 自动在反向传播时同步梯度
    6. 只在 rank 0 上保存模型和输出日志
    """
    rank, local_rank, world_size = setup_ddp()
    
    # 只在 rank 0 上输出日志（避免多个进程重复输出）
    is_main = rank == 0
    
    if is_main:
        logger.info('task: {}, model task: {}'.format(opt.task, opt.model_task))
        logger.info('Using {} GPUs for DDP training with torch.compile'.format(world_size))

    # 创建数据集
    from data.lledata import LLEData
    from data.ispdata import ISPData
    from torch.utils.data.distributed import DistributedSampler
    
    if opt.model_task == 'lle':
        dataset_class = LLEData
        train_inp_path = opt.config['train']['train_inp']
        train_gt_path = opt.config['train']['train_gt']
    elif opt.model_task == 'isp':
        dataset_class = ISPData
        train_inp_path = opt.config['train']['train_inp']
        train_gt_path = opt.config['train']['train_gt']
    else:
        raise ValueError('unknown model task')

    # K折交叉验证
    k_folds = opt.config['train'].get('k_folds', 1)
    current_fold = opt.config['train'].get('current_fold', 0)
    
    if k_folds > 1:
        if is_main:
            logger.info(f'Using {k_folds}-fold cross validation, current fold: {current_fold}')
        train_data, valid_data, n_train, n_valid = create_kfold_datasets(
            dataset_class, opt, train_inp_path, train_gt_path, k_folds, current_fold
        )
        if is_main:
            logger.info(f'Train samples: {n_train}, Valid samples: {n_valid}')
    else:
        train_data = dataset_class(opt, train_inp_path, train_gt_path)
        valid_data = dataset_class(opt, opt.config['train']['valid_inp'], opt.config['train']['valid_gt'])

    # DDP 采样器
    # DistributedSampler 的作用：
    # 1. 将数据集分成 world_size 份，每个进程处理其中一份
    # 2. shuffle=True 时，每个 epoch 会重新打乱数据（需要调用 set_epoch）
    # 3. 确保每个进程看到不同的数据，避免重复计算
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoader
    # batch_size 需要除以 world_size，因为每个进程只处理总 batch_size 的一部分
    # 例如：总 batch_size=24，3个GPU，每个GPU处理 batch_size=8
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.config['train']['batch_size'] // world_size,  # 每卡 batch_size
        sampler=train_sampler,
        num_workers=opt.config['train']['num_workers'],
        drop_last=True,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=opt.config['train']['num_workers'],
        drop_last=False,
        pin_memory=True,
    )

    # 模型
    net = import_model(opt)
    net = net.to(local_rank)
    
    # torch.compile 加速（PyTorch 2.0+）
    if opt.config['train'].get('use_compile', False) and hasattr(torch, 'compile'):
        try:
            net = torch.compile(net, mode="default")
            if is_main:
                logger.info('Using torch.compile for optimization')
        except Exception as e:
            if is_main:
                logger.info(f'torch.compile not available: {e}')
    
    # DDP 包装
    # DistributedDataParallel 的作用：
    # 1. 自动在反向传播时同步所有进程的梯度
    # 2. 梯度同步通过 all-reduce 操作实现
    # 3. find_unused_parameters=False 提高性能（如果所有参数都使用）
    # 4. gradient_as_bucket_view=True 减少内存占用
    from torch.nn.parallel import DistributedDataParallel as DDP
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False, gradient_as_bucket_view=False)
    
    if is_main:
        num_params = count_parameters(net)
        print("Total number of parameters: ", num_params)

    lr = float(opt.config['train']['lr'])
    lr_warmup = float(opt.config['train']['lr_warmup'])
    loss_warmup = import_loss('warmup')
    loss_training = import_loss(opt.model_task)

    # 检查点路径
    checkpoint_dir = opt.save_model_dir if hasattr(opt, 'save_model_dir') else './checkpoints'
    if is_main and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')

    # 判断是否断点续训
    resume_training = opt.config['model'].get('resume', False)
    start_epoch = 0
    best_psnr = 0
    best_ssim = 0
    
    # 快速指标计算器 (只在 rank 0 初始化)
    fast_metrics = None
    if is_main:
        fast_metrics = FastMetrics(device=f'cuda:{local_rank}')
    
    # EMA (Exponential Moving Average)
    ema_decay = opt.config['train'].get('ema_decay', 0.999)
    use_ema = opt.config['train'].get('use_ema', True)
    ema = None
    if use_ema and is_main:
        ema = EMA(net.module, decay=ema_decay)
        logger.info(f'EMA enabled with decay={ema_decay}')
    
    # 优化器和学习率调度器（先创建，再加载检查点）
    optim_warm = None
    optim = None
    lr_sch = None

    net.train()
    
    # Phase Warming-up
    if opt.config['train']['warmup'] and start_epoch < opt.config['train']['warmup_epoch']:
        if is_main:
            logger.info('start warming-up')

        optim_warm = torch.optim.Adam(net.parameters(), lr_warmup, weight_decay=0)
        
        # 加载检查点（如果存在）
        if resume_training and os.path.exists(checkpoint_path):
            start_epoch, best_psnr = load_checkpoint(checkpoint_path, net, optim_warm, rank=rank)
            if is_main:
                logger.info(f'Resumed from epoch {start_epoch}')
        
        epochs = opt.config['train']['warmup_epoch']
        for epo in range(start_epoch, epochs):
            # 每个 epoch 开始时调用 set_epoch，确保每个进程的数据打乱方式不同
            train_sampler.set_epoch(epo)
            loss_li = []
            for img_inp, img_gt, _ in tqdm(train_loader, ncols=80, disable=not is_main):
                # 将数据移到当前 GPU
                img_inp = img_inp.to(local_rank)
                img_gt = img_gt.to(local_rank)
                
                optim_warm.zero_grad()
                warmup_out1, warmup_out2 = net.module.forward_warm(img_inp)
                loss = loss_warmup(img_inp, img_gt, warmup_out1, warmup_out2)
                loss.backward()
                optim_warm.step()
                loss_li.append(loss.item())

            if is_main:
                avg_loss = sum(loss_li)/len(loss_li)
                logger.info('warmup epoch: {}, train_loss: {}'.format(epo+1, avg_loss))
                logger.log_scalar('warmup/train_loss', avg_loss, epo+1)
                # 保存检查点
                save_checkpoint({
                    'epoch': epo + 1,
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optim_warm.state_dict(),
                    'best_psnr': best_psnr,
                }, checkpoint_path, rank)
        
        if is_main:
            logger.info('warming-up phase done')
        
        start_epoch = 0  # warmup 结束后重置 epoch 计数

    # Phase Training
    epochs = int(opt.config['train']['epoch'])
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=0)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7)
    
    # 如果不是从 warmup 恢复，尝试加载训练检查点
    if resume_training and start_epoch == 0 and os.path.exists(checkpoint_path):
        start_epoch, best_psnr = load_checkpoint(checkpoint_path, net, optim, lr_sch, rank=rank)
        if is_main:
            logger.info(f'Resumed training from epoch {start_epoch}')

    if is_main:
        logger.info('start training')
    
    for epo in range(start_epoch, epochs):
        # 每个 epoch 开始时调用 set_epoch，确保每个进程的数据打乱方式不同
        train_sampler.set_epoch(epo)
        loss_li = []
        net.train()
        for img_inp, img_gt, _ in tqdm(train_loader, ncols=80, disable=not is_main):
            # 将数据移到当前 GPU（每个进程使用不同的 GPU）
            img_inp = img_inp.to(local_rank)
            img_gt = img_gt.to(local_rank)
            
            # 前向传播
            out = net(img_inp)
            loss = loss_training(out, img_gt)
            
            # 反向传播（DDP 自动同步所有进程的梯度）
            optim.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if opt.config['train'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.config['train']['grad_clip'])
            
            # 更新参数
            optim.step()
            
            # EMA 更新（只在 rank 0 上执行）
            if ema is not None:
                ema.update(net.module)
            loss_li.append(loss.item())
        lr_sch.step()

        # Validation (只在 rank 0 上执行)
        # 验证时只需要一个进程执行，避免重复计算
        if is_main:
            # 应用 EMA 权重进行验证（EMA 权重通常更稳定）
            if ema is not None:
                ema.apply_shadow(net.module)
            
            net.eval()
            all_ssim = []
            all_psnr = []
            
            with torch.no_grad():
                for img_inp, img_gt, _ in tqdm(valid_loader, ncols=80, disable=False, desc='Validating'):
                    img_inp = img_inp.to(local_rank)
                    img_gt = img_gt.to(local_rank)
                    
                    out = net(img_inp)
                    out = out.clamp(0, 1)
                    
                    # 计算快速指标: PSNR, SSIM
                    metrics = fast_metrics.compute(out, img_gt)
                    all_ssim.append(metrics['ssim'])
                    all_psnr.append(metrics['psnr'])
            
            mean_psnr = sum(all_psnr) / len(all_psnr)
            mean_ssim = sum(all_ssim) / len(all_ssim)
            
            # Score_fast = SSIM
            score_fast = mean_ssim
            
            current_metrics = {
                'ssim': mean_ssim,
                'psnr': mean_psnr,
                'lpips': 0.0,
                'liqe': 0.0,
                'score_fast': score_fast
            }

            train_loss = sum(loss_li) / len(loss_li)
            logger.info('epoch: {}, loss: {:.4f}, PSNR: {:.2f}, SSIM: {:.4f}'.format(
                epo+1, train_loss, mean_psnr, mean_ssim
            ))
            logger.log_scalar('train/loss', train_loss, epo+1)
            logger.log_scalar('val/psnr', mean_psnr, epo+1)
            logger.log_scalar('val/ssim', mean_ssim, epo+1)

            # 只保存SSIM最高的模型（只在 rank 0 上保存）
            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                best_psnr = mean_psnr
                
                torch.save(net.module.state_dict(), '{}/model_best.pt'.format(checkpoint_dir))
                if opt.config['train']['save_slim']:
                    net_slim = net.module.slim().to(local_rank)
                    torch.save(net_slim.state_dict(), '{}/model_best_slim.pt'.format(checkpoint_dir))
                logger.info('  -> Saved best model: model_best.pt (SSIM: {:.4f})'.format(mean_ssim))
            
            # 恢复原始权重继续训练
            if ema is not None:
                ema.restore(net.module)

    if is_main:
        logger.info('training done')
        logger.log_scalar('best_ssim', best_ssim, 0)
        logger.log_scalar('best_psnr', best_psnr, 0)
        logger.close()
    
    cleanup_ddp()


def test(opt, logger):
    """测试函数（单卡）"""
    import cv2
    
    test_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    
    psnr_list = []
    ssim_list = []
    logger.info('start testing')
    
    for (img_inp, img_gt, img_name) in test_loader:
        with torch.no_grad():
            out = net(img_inp)
            mse = ((out - img_gt)**2).mean((2, 3))
            psnr = (1 / mse).log10().mean() * 10
            
            # SSIM 计算
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            mu1 = torch.nn.functional.avg_pool2d(out, 11, stride=1, padding=5)
            mu2 = torch.nn.functional.avg_pool2d(img_gt, 11, stride=1, padding=5)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = torch.nn.functional.avg_pool2d(out * out, 11, stride=1, padding=5) - mu1_sq
            sigma2_sq = torch.nn.functional.avg_pool2d(img_gt * img_gt, 11, stride=1, padding=5) - mu2_sq
            sigma12 = torch.nn.functional.avg_pool2d(out * img_gt, 11, stride=1, padding=5) - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_val = ssim_map.mean()

        if opt.config['test']['save']:
            out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
            cv2.imwrite(r'{}/{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

        psnr_list.append(psnr.item())
        ssim_list.append(ssim_val.item())
        logger.info('image name: {}, test psnr: {:.4f}, test ssim: {:.4f}'.format(img_name[0], psnr, ssim_val))

    logger.info('testing done, overall psnr: {:.4f}, overall ssim: {:.4f}'.format(
        sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list)))


if __name__ == "__main__":
    opt = get_option()
    logger = Logger(opt)

    if opt.task == 'train':
        # 使用 torchrun 启动 DDP 训练
        train_ddp(opt, logger)
    elif opt.task == 'test':
        test(opt, logger)
    else:
        raise ValueError('unknown task, please choose from [train, test].')
