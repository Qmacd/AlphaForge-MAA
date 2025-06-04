# train_MAA.py
import torch.nn as nn
from MAA.train_multi_gan import train_multi_gan
from alphagen.models.model import ExpressionGenerator
from alphagen.data.expression import *

class AlphaDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def create_alpha_generators(n_generators, config):
    generators = []
    for _ in range(n_generators):
        generator = ExpressionGenerator(
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            d_model=config['d_model'],
            n_head=config['n_head'],
            d_ffn=config['d_ffn'],
            dropout=config['dropout'],
            operators=config['operators'],
            delta_time_range=config['delta_time_range'],
            device=config['device']
        )
        generators.append(generator)
    return generators


def create_alpha_discriminators(n_discriminators, input_dim):
    discriminators = []
    for _ in range(n_discriminators):
        discriminator = AlphaDiscriminator(input_dim)
        discriminators.append(discriminator)
    return discriminators

def calculate_ic(pred, target):
    # 计算IC
    return torch.corrcoef(torch.stack([pred, target]))[0, 1]

def calculate_rank_ic(pred, target):
    # 计算Rank IC
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))
    return calculate_ic(pred_rank, target_rank)

def validate_alpha(generator, val_x, val_y):
    generator.eval()
    with torch.no_grad():
        pred, _ = generator(val_x)
        ic = calculate_ic(pred, val_y)
        rank_ic = calculate_rank_ic(pred, val_y)
    return ic, rank_ic

def train_alpha_multi_gan(
        generators,
        discriminators,
        train_data,
        val_data,
        config,
        device
):
    # 使用你提供的train_multi_gan函数的核心逻辑
    # 但需要适配Alpha因子生成的特点

    # 1. 数据预处理
    train_xes = [prepare_data(data, window_size) for window_size in config['window_sizes']]
    train_y = prepare_target(train_data)

    # 2. 创建数据加载器
    dataloaders = create_dataloaders(train_xes, train_y, config['batch_size'])

    # 3. 设置优化器
    g_optimizers = [torch.optim.AdamW(g.parameters(), lr=config['g_lr']) for g in generators]
    d_optimizers = [torch.optim.Adam(d.parameters(), lr=config['d_lr']) for d in discriminators]

    # 4. 训练循环
    # 使用你提供的train_multi_gan函数，但需要修改：
    # - 损失函数计算方式
    # - 数据格式转换
    # - 评估指标（使用IC和Rank IC替代MSE）

    return train_multi_gan(
        generators=generators,
        discriminators=discriminators,
        dataloaders=dataloaders,
        window_sizes=config['window_sizes'],
        y_scaler=None,  # Alpha因子不需要缩放
        train_xes=train_xes,
        train_y=train_y,
        val_xes=val_xes,
        val_y=val_y,
        distill=config['use_distill'],
        cross_finetune=config['use_cross_finetune'],
        num_epochs=config['num_epochs'],
        output_dir=config['output_dir'],
        device=device,
        init_GDweight=config['init_GDweight'],
        final_GDweight=config['final_GDweight']
    )