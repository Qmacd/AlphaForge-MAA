import numpy as np
from gan.utils.builder import Builders
import torch
import torch.nn as nn
import torch.nn.functional as F
from gan.network.loss import get_losses
import copy

class NetG_DCGAN(nn.Module):
    def __init__(
            self, 
            n_chars:int,
            latent_size: int, 
            seq_len: int,
            hidden: int,
        ):
        super().__init__()
        assert seq_len == 20
        use_bias=True
        self.linear=nn.Linear(latent_size,6*384)
        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(384,256,(6,1),(1,1),bias=use_bias),#[5, 256, 47, 1]
                    nn.BatchNorm2d(256),nn.ReLU(),
                    nn.ConvTranspose2d(256,192,(5,1),(1,1),bias=use_bias),#[5, 192, 100, 1]
                    nn.BatchNorm2d(192),nn.ReLU(),
                    nn.ConvTranspose2d(192,128,(6,1),(1,1),bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                )
        self.conv=nn.Sequential(
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,128,(8,1),(1,1),0,bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,64,(8,1),(1,1),0,bias=use_bias),#[5, 64, 205, 1]
                    nn.BatchNorm2d(64),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 64, 212, 1]
                    nn.Conv2d(64,n_chars,(8,1),(1,1),0,bias=use_bias),#[5, 4, 205, 1]
        #             nn.BatchNorm2d(4)
        )
        
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0],384,6,1)
        x = self.deconv(x)
        x = self.conv(x)
        x = x.view(x.shape[0],x.shape[2],x.shape[1])
        return x #(bs, seq_len, 48)
    
class NetG_Lstm(nn.Module):
    def __init__(
        self,
        n_chars:int,
        n_layers: int,
        d_model: int,
        dropout: float,
        seq_len: int,
        potential_size: int,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.max_len = seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        
        
        self.fc_h = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.fc_c = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.emb = nn.Embedding(n_chars+1,d_model,0)
        self.rnn = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout
        )
        self.fc = nn.Linear(d_model,n_chars)
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # for name, param in self.named_parameters():
        #     nn.init.xavier_normal_(param.data)  # Initialize weight matrices using Xavier initialization
    
    def forward(self,z):
        # z: (batch_size, potential_size)
        # h,c:(n_layers, batch_size, hidden_size)
        device,bs = z.device,z.shape[0]
        h = self.fc_h(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)
        c = self.fc_c(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)
        builders = Builders(bs)
        result = []
        
        input_step = torch.full((bs,),fill_value=self.n_chars,dtype=torch.long)
        
        # onehot_s = torch.zeros(bs,self.max_len,  dtype=torch.long) 
        onehot_s = np.zeros([bs,self.max_len])
        logit_s =torch.zeros(bs,self.max_len,self.n_chars,device=device) 
        # logit_s = []
        mask_s = torch.zeros(bs,self.max_len,self.n_chars,dtype=torch.bool,device=device)
        for t in range(self.max_len):
            
            embedded = self.emb(input_step)[:,None]#[5000, 1, 128]
            if h is None:
                output,(h,c) = self.rnn(embedded)
            else:
                output,(h,c) = self.rnn(embedded,(h,c))#[2(n_layer*n_direction), bs, hidden_size]
            
            
            mask = builders.get_valid_op()# (bs, n_action)
            mask_tensor = torch.from_numpy(mask).to(device)
            logit = self.fc(output).squeeze(1) #(bs, n_chars)
            
            
            # 更新builders
            tmp = logit.detach().cpu().numpy().copy()
            tmp[~mask]=-1e8
            onehot = tmp.argmax(1)
            assert (mask[:,onehot]*1.).mean()
            builders.add_token(onehot)
            
            # 记录当前样本
            onehot_s[:,t] = onehot.flatten()
            logit_s[:,t] = logit
            # logit_s.append(self.fc(output))
            mask_s[:,t] = mask_tensor
            
            # 下一时间步输入
            input_step = torch.torch.from_numpy(onehot_s[:,t].astype(np.compat.long)).to(device)
            
            
        return (onehot_s,logit_s,mask_s),builders

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class NetG_CNN(nn.Module):
    def __init__(self, n_chars, latent_size,seq_len , hidden):
        super( ).__init__()
        self.fc1 = nn.Linear(latent_size, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size*self.seq_len, -1)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))
    
def train_network_generator(netG, netM, netP, cfg, data, target,current_round,random_method,metric,lr,n_actions):
    opt = torch.optim.Adam(netG.parameters(),lr=lr)
    best_weights = None
    best_score = -float('inf')
    patience_counter = 0
    z1 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)
    z2 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)

    netM.eval()
    netP.eval()
    
    empty_blds = None
    best_str_to_print = ''

    for epoch in range(cfg.num_epochs_g):
        netG.train()
        opt.zero_grad()
        z1 = random_method(z1)
        z2 = random_method(z2)
        logit_raw_1 =netG(z1)#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
        logit_raw_2 =netG(z2)

        masked_x_1,masks_1,blds_1= netM(logit_raw_1)
        masked_x_2,masks_2,blds_2= netM(logit_raw_2)

            
        onehot_tensor_1 = F.gumbel_softmax(masked_x_1,hard=True)
        pred_1,latent_1 = netP(onehot_tensor_1,latent=True)

        onehot_tensor_2 = F.gumbel_softmax(masked_x_2,hard=True)
        pred_2,latent_2 = netP(onehot_tensor_2,latent=True)

        loss_inputs = {
            'logit_raw_1':logit_raw_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'logit_raw_2':logit_raw_2,
            'masked_x_1':masked_x_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'masked_x_2':masked_x_2,
            'masks_1':masks_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'masks_2':masks_2,
            'blds_1':blds_1,
            'blds_2':blds_2,
            'z1':z1,#（batch_size,latent_size）
            'z2':z2,
            'onehot_tensor_1':onehot_tensor_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'onehot_tensor_2':onehot_tensor_2,
            'pred_1':pred_1,#（batch_size,1）
            'pred_2':pred_2,
            'latent_1':latent_1,#（batch_size,256）
            'latent_2':latent_2,
        }
        loss = get_losses(loss_inputs,cfg)
        
        blds:Builders = blds_1+blds_2
        idx = [i for i in range(blds.batch_size) if blds.builders[i].is_valid()]
        blds.drop_invalid()
        blds.evaluate(data,target,metric)
        
        str_to_print = f"##{epoch}/{cfg.num_epochs_g} : n_valid_train:{len(idx)}, n_valid:{len(blds.scores)}, loss:{loss:.4f}"
        mean_score = np.mean(blds.scores)
        max_score = np.max(blds.scores) if len(blds.scores)>0 else 0
        std_score = np.std(blds.scores) if len(blds.scores)>0 else 0
        str_to_print += f", max_score:{max_score:.4f},   mean_score:{mean_score:.4f}, std_score:{std_score:.4f}"
        blds.drop_duplicated()
        str_to_print += f",unique:{blds.batch_size}"
        print(str_to_print)
        if max_score>0:
            exprs = blds.exprs_str[np.argmax(blds.scores)]
            print(f"Max score {max_score} expr: {exprs}")
        # save_blds(blds,f"out/{cfg.name}/train/{current_round}",epoch)

        if empty_blds is None:
            empty_blds = blds
        else:
            empty_blds = empty_blds + blds

        
        if cfg.g_es_score == 'mean':
            es_score = mean_score
        elif cfg.g_es_score == 'max':
            es_score = max_score
        elif cfg.g_es_score == 'combined':
            es_score = max_score + 2. *  std_score
        else:
            raise NotImplementedError
        
        if es_score > best_score:
            best_score = es_score
            best_weights = copy.deepcopy(netG.state_dict())
            best_str_to_print = str_to_print
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f'Early stopping triggered at epoch {epoch}, {best_score} !')
                
                break
        
        if epoch>0:
            loss.backward()
            opt.step()

    if best_weights is not None:
        print('load_best_weights')
        netG.load_state_dict(best_weights)
        print(best_str_to_print)

    empty_blds.drop_duplicated()
    return empty_blds

def train_multi_agent_generator(netGs, netM, netP, netDs, cfg, data, target,
                                current_round, random_method, metric, lr,
                                init_weight_matrix, final_weight_matrix):
    """
    多智能体生成器+判别器对抗训练（融合蒸馏 + 判别器矩阵 + Cross Finetune）
    返回值与 train_network_generator 一致：合并后的 Builders
    """
    N = len(netGs)
    optGs = [torch.optim.Adam(netG.parameters(), lr=lr) for netG in netGs]
    optDs = [torch.optim.Adam(netD.parameters(), lr=lr) for netD in netDs]
    best_scores = [-float('inf')] * N
    best_weights = [None] * N
    patience_counter = 0
    merged_blds = None

    for epoch in range(cfg.num_epochs_g):
        weight_matrix = torch.tensor(
            init_weight_matrix if epoch < 20 else final_weight_matrix
        ).to(cfg.device)

        z_list = [random_method(torch.zeros(cfg.batch_size, cfg.potential_size).to(cfg.device)) for _ in range(N)]
        logits_list = [netG(z) for netG, z in zip(netGs, z_list)]

        netM.eval()
        netP.eval()
        for netD in netDs:
            netD.train()

        masked_x_list, onehot_list, blds_list = [], [], []

        for logit in logits_list:
            masked_x, masks, blds = netM(logit)
            onehot = F.gumbel_softmax(masked_x, hard=True)
            masked_x_list.append(masked_x)
            onehot_list.append(onehot)

            blds.evaluate(data, target, metric)
            blds.drop_invalid()
            blds.drop_duplicated()
            blds_list.append(blds)

        # 判别器监督
        gan_loss_matrix = torch.zeros(N, N).to(cfg.device)
        for i in range(N):
            for j in range(N):
                fake_input = onehot_list[i].detach().view(onehot_list[i].size(0), -1)
                score = netDs[j](fake_input)
                real_label = torch.ones_like(score).to(cfg.device)
                gan_loss_matrix[j, i] = F.binary_cross_entropy(score, real_label)

        loss_G_all = (weight_matrix[:, :-1] * gan_loss_matrix).sum(dim=1)

        # 不使用 pred_loss，直接使用对抗损失（兼容原始逻辑）
        loss_total_list = [loss_G_all[i] for i in range(N)]

        for optG in optGs:
            optG.zero_grad()
        for loss in loss_total_list:
            loss.backward(retain_graph=True)
        for optG in optGs:
            optG.step()

        # 蒸馏
        if epoch % 10 == 0 and cfg.distill:
            with torch.no_grad():
                score_arr = [np.mean(b.scores) if len(b.scores) > 0 else -float('inf') for b in blds_list]

            # 确保有有效分数
            valid_scores = [s for s in score_arr if s != -float('inf')]
            if len(valid_scores) < 2:
                continue

            rank = np.argsort(score_arr)[::-1]
            teacher_idx, student_idx = rank[0], rank[-1]

            # 如果教师和学生是同一个，跳过
            if teacher_idx == student_idx:
                continue

            try:
                # 重新生成用于蒸馏的输入
                z_teacher = random_method(torch.zeros(cfg.batch_size, cfg.potential_size).to(cfg.device))
                z_student = random_method(torch.zeros(cfg.batch_size, cfg.potential_size).to(cfg.device))

                with torch.no_grad():
                    teacher_logits = netGs[teacher_idx](z_teacher)

                student_logits = netGs[student_idx](z_student)

                # 计算蒸馏损失
                soft_loss = distill_step(student_logits, teacher_logits, temperature=2.0)

                # 优化学生模型
                optGs[student_idx].zero_grad()
                soft_loss.backward()
                optGs[student_idx].step()

            except Exception as e:
                print(f"Distillation failed at epoch {epoch}: {e}")
                continue

        # Cross Finetune 判别器
        if epoch % 10 == 0 and cfg.cross_finetune:
            try:
                i, j = np.random.choice(N, 2, replace=False)

                # 重新生成数据，避免使用已经参与梯度计算的数据
                with torch.no_grad():
                    z_new = random_method(torch.zeros(cfg.batch_size, cfg.potential_size).to(cfg.device))
                    logit_new = netGs[i](z_new)
                    masked_x_new, _, _ = netM(logit_new)
                    onehot_new = F.gumbel_softmax(masked_x_new, hard=True)

                # 用新数据训练判别器
                netDs[j].train()
                netGs[i].eval()  # 固定生成器

                fake_input = onehot_new.view(onehot_new.size(0), -1).detach()
                out = netDs[j](fake_input)

                # 判别器希望把假数据识别为假(0)，但这里是对抗训练，所以用1
                real_label = torch.ones_like(out).to(cfg.device)
                loss_cross = F.binary_cross_entropy_with_logits(out, real_label)

                optDs[j].zero_grad()
                loss_cross.backward()
                optDs[j].step()

            except Exception as e:
                print(f"Cross finetune failed at epoch {epoch}: {e}")
                continue

        # Early stopping 逻辑
        avg_scores = [np.mean(b.scores) for b in blds_list]
        if max(avg_scores) > max(best_scores):
            for i in range(N):
                if avg_scores[i] > best_scores[i]:
                    best_scores[i] = avg_scores[i]
                    best_weights[i] = copy.deepcopy(netGs[i].state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f"Early stopping at epoch {epoch}")
                break

        # 合并 blds
        if merged_blds is None:
            merged_blds = copy.deepcopy(blds_list[0])
            for b in blds_list[1:]:
                merged_blds += copy.deepcopy(b)
        else:
            for b in blds_list:
                merged_blds += copy.deepcopy(b)
            for i, b in enumerate(blds_list):
                print(f"[Agent {i}] valid builders: {b.batch_size}, avg score: {np.mean(b.scores):.4f}")
    # 加载最优权重
    for i in range(N):
        if best_weights[i] is not None:
            netGs[i].load_state_dict(best_weights[i])

    merged_blds.drop_duplicated()
    return merged_blds


def distill_step(student_logits, teacher_logits, temperature=1.0):
    """蒸馏损失计算"""
    # 确保维度一致
    if student_logits.shape != teacher_logits.shape:
        print(f"Shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}")
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # 教师输出需要detach
    teacher_logits = teacher_logits.detach()

    # 温度软化
    teacher_softmax = F.softmax(teacher_logits / temperature, dim=-1)
    student_logsoftmax = F.log_softmax(student_logits / temperature, dim=-1)

    # KL散度损失
    distillation_loss = F.kl_div(student_logsoftmax, teacher_softmax, reduction='batchmean')

    return distillation_loss
