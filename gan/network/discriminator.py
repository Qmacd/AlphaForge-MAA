import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(Discriminator, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 时序特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # 对抗分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 * seq_len, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 组件对抗器
        self.component_adversary = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 特征提取
        features = self.feature_extractor(x.view(-1, self.input_dim))
        features = features.view(batch_size, self.seq_len, -1)
        
        # 时序特征提取
        temporal_features = self.temporal_conv(features.transpose(1, 2))
        temporal_features = temporal_features.transpose(1, 2)
        
        # 全局判别
        global_features = temporal_features.reshape(batch_size, -1)
        global_output = self.classifier(global_features)
        
        # 组件对抗
        component_outputs = []
        for t in range(self.seq_len):
            component_output = self.component_adversary(temporal_features[:, t])
            component_outputs.append(component_output)
        component_outputs = torch.stack(component_outputs, dim=1)
        
        return global_output, component_outputs 