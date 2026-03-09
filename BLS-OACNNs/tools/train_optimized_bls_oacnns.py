#!/usr/bin/env python3
"""
BLS-OACNNs优化版训练脚本
P0阶段优化训练工具

使用方法:
python tools/train_optimized_bls_oacnns.py --config-file configs/scannet/semseg-bls-oacnns-optimized-v1m1-0-base.py
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from pointcept.models.bls_oacnns_optimized import Optimized_BLS_OACNNs
from pointcept.datasets import build_dataset
from pointcept.engines import default_trainer
from pointcept.utils.misc import get_dist_info, init_random_seed


class OptimizedBLSTrainer:
    """
    优化版BLS-OACNNs训练器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        seed = config.get('random_seed', 42)
        init_random_seed(seed)

        # 初始化模型
        self.model = self._build_model()
        self.model.to(self.device)

        # 初始化数据集
        self.train_dataset, self.val_dataset = self._build_datasets()

        # 初始化数据加载器
        self.train_loader = self._build_dataloader(self.train_dataset, training=True)
        self.val_loader = self._build_dataloader(self.val_dataset, training=False)

        # 初始化优化器和调度器
        self.optimizer, self.scheduler, self.lr_monitor = self.model.get_optimizer_and_scheduler(
            learning_rate=config['optimizer']['lr'],
            scheduler_type=config.get('scheduler_type', 'adaptive')
        )

        # 初始化损失函数
        self.criterion = self._build_criterion()

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []

        # 日志设置
        self._setup_logging()

        self.logger.info("Optimized BLS-OACNNs trainer initialized successfully")
        self.logger.info(f"Model config: stem_type={getattr(self.model, 'stem_type', 'unknown')}, "
                       f"output_head_type={getattr(self.model, 'output_head_type', 'unknown')}, "
                       f"bls_ratio={getattr(self.model, 'bls_ratio', 0.6)}")

    def _build_model(self):
        """构建模型"""
        model = Optimized_BLS_OACNNs(**self.config['model'])

        # 数据并行
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

        return model

    def _build_datasets(self):
        """构建数据集"""
        train_dataset = build_dataset(
            self.config['dataset'],
            split='train',
            transform=self.config.get('train_transform', None)
        )

        val_dataset = build_dataset(
            self.config['dataset'],
            split='val',
            transform=self.config.get('val_transform', None)
        )

        return train_dataset, val_dataset

    def _build_dataloader(self, dataset, training=True):
        """构建数据加载器"""
        batch_size = self.config['batch_size'] if training else self.config.get('val_batch_size', self.config['batch_size'])
        num_workers = self.config.get('num_workers', 4)

        shuffle = training
        drop_last = training

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def _build_criterion(self):
        """构建损失函数"""
        if hasattr(self.model, 'auxiliary_loss') and self.model.auxiliary_loss:
            return self.model.auxiliary_loss
        else:
            return nn.CrossEntropyLoss(ignore_index=-1)

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 文件处理器
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / f'train_{int(time.time())}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(self.train_loader):
            # 数据转移到设备
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}

            # 前向传播
            self.optimizer.zero_grad()

            if self.model.use_auxiliary_loss and hasattr(self.model, 'output_head_type'):
                if self.model.output_head_type == 'progressive':
                    output = self.model(batch_data, return_aux=True)
                    if isinstance(output, tuple):
                        main_output, aux_output = output
                        loss, main_loss, aux_loss = self.criterion(main_output, aux_output, batch_data['label'])
                    else:
                        output = self.model(batch_data)
                        loss = self.criterion(output, batch_data['label'])
                        main_loss = loss
                        aux_loss = torch.tensor(0.0, device=loss.device)
                else:
                    output = self.model(batch_data)
                    loss = self.criterion(output, batch_data['label'])
                    main_loss = loss
                    aux_loss = torch.tensor(0.0, device=loss.device)
            else:
                output = self.model(batch_data)
                loss = self.criterion(output, batch_data['label'])
                main_loss = loss
                aux_loss = torch.tensor(0.0, device=loss.device)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # 日志记录
            if batch_idx % 50 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                               f"Loss={loss.item():.4f}, Main={main_loss.item():.4f}, Aux={aux_loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                # 数据转移到设备
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}

                # 前向传播
                output = self.model(batch_data)
                loss = self.criterion(output, batch_data['label'])

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # 更新学习率监控
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_monitor.update(epoch, current_lr, avg_loss, 0.0)  # 这里可以添加实际指标

        return avg_loss

    def train(self):
        """完整训练流程"""
        max_epochs = self.config['max_epochs']

        self.logger.info(f"Starting training for {max_epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate_epoch(epoch)

            # 学习率调度
            self.scheduler.step()

            # 日志记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                           f"Val Loss={val_loss:.4f}, LR={current_lr:.6f}")

            # 保存最佳模型
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self._save_checkpoint(epoch, is_best=True)

            # 定期保存检查点
            if (epoch + 1) % 100 == 0:
                self._save_checkpoint(epoch)

            # 学习率监控
            if (epoch + 1) % 50 == 0:
                self.lr_monitor.plot_lr_schedule()

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_metric:.4f}")

    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config,
        }

        # 保存目录
        save_dir = Path('checkpoints')
        save_dir.mkdir(exist_ok=True)

        if is_best:
            torch.save(checkpoint, save_dir / 'best_model.pth')
            self.logger.info(f"Saved best model at epoch {epoch}")
        else:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
            self.logger.info(f"Saved checkpoint at epoch {epoch}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Optimized BLS-OACNNs')
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    import yaml
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    if args.seed:
        config['random_seed'] = args.seed

    # 创建训练器
    trainer = OptimizedBLSTrainer(config)

    # 恢复训练
    if args.resume:
        trainer._load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()