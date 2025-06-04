import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_generator_losses(data_G, output_dir):
    plt.figure(figsize=(12, 6))
    for i, losses in enumerate(data_G):
        plt.plot(losses[-1], label=f'G{i+1}')
    plt.title('Generator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'generator_losses.png'))
    plt.close()

def plot_discriminator_losses(data_D, output_dir):
    plt.figure(figsize=(12, 6))
    for i, losses in enumerate(data_D):
        plt.plot(losses[-1], label=f'D{i+1}')
    plt.title('Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'discriminator_losses.png'))
    plt.close()

def visualize_overall_loss(g_losses, d_losses, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(g_losses, label='Generator')
    plt.plot(d_losses, label='Discriminator')
    plt.title('Overall GAN Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'overall_losses.png'))
    plt.close()

def plot_mse_loss(mse_losses, val_losses, epoch, output_dir):
    plt.figure(figsize=(12, 6))
    for i, (mse, val) in enumerate(zip(mse_losses, val_losses)):
        plt.plot(mse[:epoch], label=f'G{i+1} MSE')
        plt.plot(val[:epoch], label=f'G{i+1} Val')
    plt.title('MSE and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mse_val_losses.png'))
    plt.close()

def evaluate_best_models(generators, best_model_state, train_xes, train_y, val_xes, val_y, y_scaler, output_dir):
    results = {}
    
    for i, (generator, state_dict) in enumerate(zip(generators, best_model_state)):
        generator.load_state_dict(state_dict)
        generator.eval()
        
        with torch.no_grad():
            # 训练集评估
            train_pred, train_cls = generator(train_xes[i])
            train_mse = torch.mean((train_pred - train_y[i:i+len(train_xes[i])]) ** 2).item()
            
            # 验证集评估
            val_pred, val_cls = generator(val_xes[i])
            val_mse = torch.mean((val_pred - val_y[i:i+len(val_xes[i])]) ** 2).item()
            
            # 计算分类准确率
            train_acc = (train_cls.argmax(dim=1) == train_y[i:i+len(train_xes[i])].argmax(dim=1)).float().mean().item()
            val_acc = (val_cls.argmax(dim=1) == val_y[i:i+len(val_xes[i])].argmax(dim=1)).float().mean().item()
        
        results[f'G{i+1}'] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    
    return results 