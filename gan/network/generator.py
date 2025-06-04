import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, seq_len):
        super(Generator, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Masker网络
        self.masker = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出mask值在0-1之间
        )
        
        # DCGAN生成器主体
        self.main = nn.Sequential(
            # 输入层
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            # 隐藏层1
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # 隐藏层2
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # 输出层
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # 分类器分支
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 生成mask
        mask = self.masker(x.view(-1, self.input_dim))
        mask = mask.view(batch_size, self.seq_len, self.input_dim)
        
        # 应用mask
        masked_x = x * mask
        
        # 生成器输出
        gen_output = self.main(masked_x.view(-1, self.input_dim))
        gen_output = gen_output.view(batch_size, self.seq_len, -1)
        
        # 分类器输出
        cls_output = self.classifier(masked_x.view(-1, self.input_dim))
        cls_output = cls_output.view(batch_size, self.seq_len, -1)
        
        return gen_output, cls_output, mask 