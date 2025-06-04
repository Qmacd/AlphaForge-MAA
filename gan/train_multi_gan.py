import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler
import time
import logging

from utils.evaluate_visualization import (
    plot_generator_losses,
    plot_discriminator_losses,
    visualize_overall_loss,
    plot_mse_loss,
    evaluate_best_models
)

def train_multi_gan(generators, discriminators, dataloaders,
                    window_sizes,
                    y_scaler, train_xes, train_y, val_xes, val_y,
                    distill, cross_finetune,
                    num_epochs,
                    output_dir,
                    device,
                    init_GDweight=[
                        [1, 0, 0, 1.0],  # alphas_init
                        [0, 1, 0, 1.0],  # betas_init
                        [0., 0, 1, 1.0]  # gammas_init...
                    ],
                    final_GDweight=[
                        [0.333, 0.333, 0.333, 1.0],  # alphas_final
                        [0.333, 0.333, 0.333, 1.0],  # betas_final
                        [0.333, 0.333, 0.333, 1.0]  # gammas_final...
                    ],
                    logger=None):
    N = len(generators)

    assert N == len(discriminators)
    assert N == len(window_sizes)
    assert N > 1

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5

    criterion = nn.BCELoss()

    optimizers_G = [torch.optim.AdamW(model.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    for model in generators]

    schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, min_lr=1e-7)
                  for optimizer in optimizers_G]

    optimizers_D = [torch.optim.Adam(model.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
                    for model in discriminators]

    best_epoch = [-1 for _ in range(N)]

    keys = []
    g_keys = [f'G{i}' for i in range(1, N + 1)]
    d_keys = [f'D{i}' for i in range(1, N + 1)]
    MSE_g_keys = [f'MSE_G{i}' for i in range(1, N + 1)]
    val_loss_keys = [f'val_G{i}' for i in range(1, N + 1)]

    keys.extend(g_keys)
    keys.extend(d_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)

    d_g_keys = []
    for g_key in g_keys:
        for d_key in d_keys:
            d_g_keys.append(d_key + "_" + g_key)
    keys.extend(d_g_keys)

    hists_dict = {key: np.zeros(num_epochs) for key in keys}

    best_mse = [float('inf') for _ in range(N)]
    best_model_state = [None for _ in range(N)]

    patience_counter = 0
    patience = 15
    feature_num = train_xes[0].shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    for epoch in range(num_epochs):
        epo_start = time.time()

        if epoch < 20:
            weight_matrix = torch.tensor(init_GDweight).to(device)
        else:
            weight_matrix = torch.tensor(final_GDweight).to(device)

        keys = []
        keys.extend(g_keys)
        keys.extend(d_keys)
        keys.extend(MSE_g_keys)
        keys.extend(d_g_keys)

        loss_dict = {key: [] for key in keys}

        gaps = [window_sizes[-1] - window_sizes[i] for i in range(N - 1)]

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloaders[-1]):
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)

            X = []
            Y = []
            LABELS = []

            for gap in gaps:
                X.append(x_last[:, gap:, :])
                Y.append(y_last[:, gap:, :])
                LABELS.append(label_last[:, gap:, :])
            X.append(x_last.to(device))
            Y.append(y_last.to(device))
            LABELS.append(label_last.to(device))

            for i in range(N):
                generators[i].eval()
                discriminators[i].train()

            loss_D, lossD_G = discriminate_fake(X, Y, LABELS,
                                                generators, discriminators,
                                                window_sizes, target_num,
                                                criterion, weight_matrix,
                                                device, mode="train_D")

            for i in range(N):
                loss_dict[d_keys[i]].append(loss_D[i].item())

            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    key = f'D{i}_G{j}'
                    loss_dict[key].append(lossD_G[i - 1, j - 1].item())

            for optimizer_D in optimizers_D:
                optimizer_D.zero_grad()

            loss_D.sum(dim=0).backward()

            for i in range(N):
                optimizers_D[i].step()
                discriminators[i].eval()
                generators[i].train()

            weight = weight_matrix[:, :-1].clone().detach()

            loss_G, loss_mse_G = discriminate_fake(X, Y, LABELS,
                                                   generators, discriminators,
                                                   window_sizes, target_num,
                                                   criterion, weight,
                                                   device,
                                                   mode="train_G")

            for i in range(N):
                loss_dict[g_keys[i]].append(loss_G[i].item())
                loss_dict["MSE_" + g_keys[i]].append(loss_mse_G[i].item())

            for optimizer_G in optimizers_G:
                optimizer_G.zero_grad()

            loss_G.sum(dim=0).backward()

            for optimizer_G in optimizers_G:
                optimizer_G.step()

        for key in loss_dict.keys():
            hists_dict[key][epoch] = np.mean(loss_dict[key])

        improved = [False] * 3
        for i in range(N):
            hists_dict[val_loss_keys[i]][epoch] = validate(generators[i], val_xes[i], val_y)

            if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
                best_mse[i] = hists_dict[val_loss_keys[i]][epoch]
                best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                best_epoch[i] = epoch + 1
                improved[i] = True

            schedulers[i].step(hists_dict[val_loss_keys[i]][epoch])

        if distill and epoch+1 % 10 == 0:
            losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            rank = np.argsort(losses)
            print("Do distill one epoch!")
            do_distill(rank, generators, dataloaders, optimizers_G, window_sizes, device)
            
        if epoch+1 % 10 == 0 and cross_finetune:
            G_losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            D_losses = [np.mean(loss_dict[d_keys[i]]) for i in range(N)]
            G_rank = np.argsort(G_losses)
            D_rank = np.argsort(D_losses)
            print(f"Start cross finetune!  G{G_rank[0]+1} with D{D_rank[0]+1}")
            print()
            
            for e in range(5):
                cross_best_Gloss = np.inf

                generators[G_rank[0]].eval()
                discriminators[D_rank[0]].train()

                loss_D, lossD_G = discriminate_fake([X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[G_rank[0]]],
                                                    [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                    [window_sizes[D_rank[0]]], target_num,
                                                    criterion, weight_matrix[D_rank[0], G_rank[0]],
                                                    device, mode="train_D")

                optimizers_D[D_rank[0]].zero_grad()
                loss_D.sum(dim=0).backward()
                optimizers_D[D_rank[0]].step()
                discriminators[D_rank[0]].eval()
                generators[G_rank[0]].train()

                weight = weight_matrix[:, :-1].clone().detach()
                loss_G, loss_mse_G = discriminate_fake([X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[G_rank[0]]],
                                                       [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                       [window_sizes[D_rank[0]]], target_num,
                                                       criterion, weight[D_rank[0], G_rank[0]],
                                                       device,
                                                       mode="train_G")

                optimizers_G[G_rank[0]].zero_grad()
                loss_G.sum(dim=0).backward()
                optimizers_G[G_rank[0]].step()

                validate_G_loss = validate(generators[G_rank[0]], val_xes[G_rank[0]], val_y)

                if validate_G_loss >= cross_best_Gloss:
                    generators[G_rank[0]].load_state_dict(best_model_state[G_rank[0]])
                    break
                elif validate_G_loss < cross_best_Gloss:
                    cross_best_Gloss = validate_G_loss
                    best_mse[G_rank[0]] = cross_best_Gloss
                    best_model_state[G_rank[0]] = copy.deepcopy(generators[G_rank[0]].state_dict())
                    best_epoch[G_rank[0]] = epoch + 1

                print(f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}")
                logging.info(f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        log_str = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]:.8f}"
            for i, key in enumerate(val_loss_keys)
        )
        print(f"patience counter:{patience_counter}")
        logging.info("EPOCH %d | Validation MSE: %s ", epoch + 1, log_str)
        
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        epo_end = time.time()
        print(f"Epoch time: {epo_end - epo_start:.4f}")

    data_G = [[[] for _ in range(4)] for _ in range(N)]
    data_D = [[[] for _ in range(4)] for _ in range(N)]

    for i in range(N):
        for j in range(N + 1):
            if j < N:
                data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch]
                data_D[i][j] = hists_dict[f"D{i + 1}_G{j + 1}"][:epoch]
            elif j == N:
                data_G[i][j] = hists_dict[g_keys[i]][:epoch]
                data_D[i][j] = hists_dict[d_keys[i]][:epoch]

    plot_generator_losses(data_G, output_dir)
    plot_discriminator_losses(data_D, output_dir)
    visualize_overall_loss([data_G[i][N] for i in range(N)], [data_D[i][N] for i in range(N)], output_dir)

    hist_MSE_G = [[] for _ in range(N)]
    hist_val_loss = [[] for _ in range(N)]
    for i in range(N):
        hist_MSE_G[i] = hists_dict[f"MSE_G{i + 1}"][:epoch]
        hist_val_loss[i] = hists_dict[f"val_G{i + 1}"][:epoch]

    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch, output_dir)

    for i in range(N):
        print(f"G{i + 1} best epoch: ", best_epoch[i])
        logging.info(f"G{i + 1} best epoch: {best_epoch[i]}")

    results = evaluate_best_models(generators, best_model_state, train_xes, train_y, val_xes, val_y, y_scaler,
                                   output_dir)

    return results, best_model_state

def discriminate_fake(X, Y, LABELS,
                      generators, discriminators,
                      window_sizes, target_num,
                      criterion, weight_matrix,
                      device,
                      mode):
    assert mode in ["train_D", "train_G"]

    N = len(generators)

    dis_real_outputs = [model(y) for (model, y) in zip(discriminators, Y)]
    real_labels = [torch.ones_like(dis_real_output).to(device) for dis_real_output in dis_real_outputs]
    outputs = [generator(x) for (generator, x) in zip(generators, X)]
    fake_data_G, fake_data_cls = zip(*outputs)

    lossD_real = [criterion(dis_real_output, real_label) for (dis_real_output, real_label) in
                  zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        fake_data_temp_G = [fake_data.detach() for fake_data in fake_data_G]
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_temp_G)]
    elif mode == "train_G":
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_G)]

    fake_data_GtoD = {}
    for i in range(N):
        for j in range(N):
            if i < j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :window_sizes[j] - window_sizes[i], :], fake_data_temp_G[i]], axis=1)
            elif i > j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
            elif i == j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i]

    fake_labels = [torch.zeros_like(real_label).to(device) for real_label in real_labels]

    dis_fake_outputD = []
    for i in range(N):
        row = []
        for j in range(N):
            out = discriminators[i](fake_data_GtoD[f"G{j + 1}ToD{i + 1}"])
            row.append(out)
        if mode == "train_D":
            row.append(lossD_real[i])
        dis_fake_outputD.append(row)

    if mode == "train_D":
        loss_matrix = torch.zeros(N, N + 1, device=device)
        weight = weight_matrix.clone().detach()
        for i in range(N):
            for j in range(N + 1):
                if j < N:
                    loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], fake_labels[i])
                elif j == N:
                    loss_matrix[i, j] = dis_fake_outputD[i][j]
    elif mode == "train_G":
        loss_matrix = torch.zeros(N, N, device=device)
        weight = weight_matrix.clone().detach()
        for i in range(N):
            for j in range(N):
                loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], real_labels[i])

    loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)

    if mode == "train_G":
        loss_mse_G = [F.mse_loss(fake_data.squeeze(), y[:, -1, :].squeeze()) for (fake_data, y) in zip(fake_data_G, Y)]
        loss_matrix = loss_mse_G
        loss_DorG = loss_DorG + torch.stack(loss_matrix).to(device)
        
        cls_losses = [F.cross_entropy(fake_cls, l[:, -1, :].long().squeeze()) for (fake_cls, l) in
                      zip(fake_data_cls, LABELS)]
        classification_loss = torch.stack(cls_losses)
        loss_DorG = loss_DorG + classification_loss

    return loss_DorG, loss_matrix

def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device,
               *,
               alpha: float = 0.7,
               temperature: float = 2.0,
               grad_clip: float = 1.0,
               mse_lambda: float = 0.5,
               ):
    teacher_generator = generators[rank[0]]
    student_generator = generators[rank[-1]]
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    
    for batch_idx, (x, y, label) in enumerate(distill_dataloader):
        y = y[:, -1, :]
        y = y.to(device)
        label = label[:, -1]
        label = label.to(device)
        
        if gap > 0:
            x_teacher = x
            x_student = x[:, gap:, :]
        else:
            x_teacher = x[:, (-1) * gap:, :]
            x_student = x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        teacher_output, teacher_cls = teacher_generator(x_teacher)
        teacher_output, teacher_cls = teacher_output.detach(), teacher_cls.detach()
        student_output, student_cls = student_generator(x_student)

        teacher_soft = F.softmax(teacher_cls.detach() / temperature, dim=1)
        student_log_soft = F.log_softmax(student_cls / temperature, dim=1)

        soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (alpha * temperature ** 2)

        hard_loss = F.cross_entropy(student_cls, label.long()) * (1 - alpha)
        hard_loss += F.mse_loss(student_output * temperature, y) * (1 - alpha) * mse_lambda
        distillation_loss = soft_loss + hard_loss

        student_optimizer.zero_grad()
        distillation_loss.backward()

        if grad_clip is not None:
            clip_grad_norm_(student_generator.parameters(), grad_clip)

        student_optimizer.step() 