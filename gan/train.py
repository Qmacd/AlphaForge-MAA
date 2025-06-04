import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging
from datetime import datetime

from network.generator import Generator
from network.discriminator import Discriminator
from train_multi_gan import train_multi_gan

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, target_cols, feature_cols):
        self.data = data
        self.window_size = window_size
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        x = window[self.feature_cols].values
        y = window[self.target_cols].values
        labels = window['label'].values if 'label' in window.columns else np.zeros(len(window))
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor(labels)

def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def prepare_data(csv_path, window_sizes, batch_size=32, target_cols=None, feature_cols=None):
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 如果没有指定目标列和特征列，使用所有数值列作为特征
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_cols is not None:
            feature_cols = [col for col in feature_cols if col not in target_cols]
    
    if target_cols is None:
        target_cols = feature_cols
    
    # 数据标准化
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 准备不同窗口大小的数据
    train_xes = []
    val_xes = []
    train_y = None
    val_y = None
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # 创建数据加载器
    dataloaders = []
    for window_size in window_sizes:
        # 训练集
        train_dataset = TimeSeriesDataset(
            train_df, window_size, target_cols, feature_cols
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        dataloaders.append(train_loader)
        
        # 准备训练和验证数据
        train_x = torch.stack([x for x, _, _ in train_dataset])
        train_y = torch.stack([y for _, y, _ in train_dataset])
        
        val_dataset = TimeSeriesDataset(
            val_df, window_size, target_cols, feature_cols
        )
        val_x = torch.stack([x for x, _, _ in val_dataset])
        val_y = torch.stack([y for _, y, _ in val_dataset])
        
        train_xes.append(train_x)
        val_xes.append(val_x)
    
    return dataloaders, train_xes, train_y, val_xes, val_y, scaler

def main():
    # 配置参数
    config = {
        'csv_path': 'path/to/your/data.csv',  # 替换为您的CSV文件路径
        'output_dir': 'outputs/multi_gan',
        'window_sizes': [5, 10, 15],  # 不同窗口大小
        'batch_size': 32,
        'num_epochs': 100,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'feature_dim': 10,  # 特征维度
        'hidden_dim': 64,   # 隐藏层维度
        'num_classes': 2,   # 分类类别数
        'target_cols': ['target1', 'target2'],  # 替换为您的目标列名
        'feature_cols': ['feature1', 'feature2', 'feature3']  # 替换为您的特征列名
    }
    
    # 设置日志
    setup_logging(config['output_dir'])
    
    # 准备数据
    dataloaders, train_xes, train_y, val_xes, val_y, scaler = prepare_data(
        config['csv_path'],
        config['window_sizes'],
        config['batch_size'],
        config['target_cols'],
        config['feature_cols']
    )
    
    # 初始化生成器和判别器
    generators = []
    discriminators = []
    
    for window_size in config['window_sizes']:
        # 生成器
        generator = Generator(
            input_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['feature_dim'],
            num_classes=config['num_classes'],
            seq_len=window_size
        ).to(config['device'])
        generators.append(generator)
        
        # 判别器
        discriminator = Discriminator(
            input_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            seq_len=window_size
        ).to(config['device'])
        discriminators.append(discriminator)
    
    # 训练模型
    results, best_model_state = train_multi_gan(
        generators=generators,
        discriminators=discriminators,
        dataloaders=dataloaders,
        window_sizes=config['window_sizes'],
        y_scaler=scaler,
        train_xes=train_xes,
        train_y=train_y,
        val_xes=val_xes,
        val_y=val_y,
        distill=True,
        cross_finetune=True,
        num_epochs=config['num_epochs'],
        output_dir=config['output_dir'],
        device=config['device']
    )
    
    # 保存最佳模型
    for i, state_dict in enumerate(best_model_state):
        torch.save(state_dict, os.path.join(config['output_dir'], f'best_generator_{i+1}.pth'))
    
    logging.info("Training completed!")
    logging.info(f"Results: {results}")

if __name__ == '__main__':
    main() 