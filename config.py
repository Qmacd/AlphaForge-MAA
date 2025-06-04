# config.py
from alphagen.data.expression import Operators

alpha_gan_config = {
    'n_generators': 3,
    'n_discriminators': 3,
    'window_sizes': [5, 10, 20],  # 不同时间窗口
    'n_encoder_layers': 6,
    'n_decoder_layers': 6,
    'd_model': 512,
    'n_head': 8,
    'd_ffn': 2048,
    'dropout': 0.1,
    'operators': [op for op in Operators],  # 使用现有的操作符
    'delta_time_range': (-20, 20),
    'device': 'cuda:0',
    'batch_size': 32,
    'g_lr': 2e-5,
    'd_lr': 2e-5,
    'num_epochs': 100,
    'use_distill': True,
    'use_cross_finetune': True,
    'output_dir': 'outputs/alpha_gan',
    'init_GDweight': [
        [1, 0, 0, 1.0],
        [0, 1, 0, 1.0],
        [0, 0, 1, 1.0]
    ],
    'final_GDweight': [
        [0.333, 0.333, 0.333, 1.0],
        [0.333, 0.333, 0.333, 1.0],
        [0.333, 0.333, 0.333, 1.0]
    ]
}