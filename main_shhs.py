# nohup python3 main_edf78.py &
import os
import numpy as np
import torch
from utils import load_train_dataset, load_dataset, load_dataloader, mean_std
from structure.mass import MASS
from torch.optim import lr_scheduler
import math
import torch.nn as nn
import random
from collections import defaultdict
import logging
import time
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
from lion_pytorch import Lion
from train import classifier_train
import argparse

def set_seed(seed=2025):
    """
    设置所有相关库的随机种子以确保结果可复现
    Args:
        seed: 随机种子值，默认为2025
    """
    random.seed(seed)  # Python随机数生成器
    np.random.seed(seed)  # NumPy随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机数生成器
    torch.cuda.manual_seed_all(seed)  # 所有GPU的随机数生成器
    # 确保卷积算法的确定性，提高结果可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CosineAnnealingWarmupRestarts(lr_scheduler._LRScheduler):
    """
    带有warmup的cosine学习率调度器
    Args:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        warmup_steps: warmup阶段的步数
        max_lr: 最大学习率
        min_lr: 最小学习率
        gamma: 每个周期后的衰减因子
    """
    def __init__(self, optimizer, first_cycle_steps, warmup_steps=0, max_lr=1e-3, min_lr=1e-6, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps  # 第一个周期的总步数
        self.warmup_steps = warmup_steps  # warmup步数
        self.max_lr = max_lr  # 最大学习率
        self.min_lr = min_lr  # 最小学习率
        self.gamma = gamma  # 衰减因子
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup阶段：线性增长从0到max_lr
            return [self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps 
                   for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
            return [self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                   for _ in self.optimizer.param_groups]

# model parameters - 模型结构参数
embed_dim = 128  # 嵌入维度
num_heads = 8  # 多头注意力头数
mlp_ratio = 4  # MLP隐藏层维度倍数
dropout = 0.1  # dropout概率，增加正则化
num_patches = 30  # patch数量
num_epoches = 32  # 每个样本的epoch数
global_encoder_depth = 4  # 全局编码器深度，增加模型容量
intra_encoder_depth = 4  # intra编码器深度
inter_encoder_depth = 2  # inter编码器深度
inter_hidden_dim = 256  # inter隐藏层维度
step = 32  # 步长参数

# train parameters - 训练参数
params = {
    "lr": 1e-4,  # 初始学习率
    "weight_decay": 0.01,  # 权重衰减
    "total_epochs": 100,  # 总训练轮数
    "warmup_epochs": 10,  # warmup轮数（前10%）
    "min_lr": 1e-6,  # 最小学习率
}


def get_all_files(folder_path):
    """
    获取数据集中所有文件路径
    每个文件对应一个独立的受试者（329个文件，329个受试者）
    Returns:
        files: 所有.mat文件的完整路径列表
    """
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            files.append(os.path.join(folder_path, file))
    return sorted(files)  # 排序确保一致性


def create_folds(files, num_folds=20, random_seed=2025):
    """
    创建交叉验证的fold划分
    Args:
        files: 所有文件路径列表
        num_folds: fold数量，默认20
        random_seed: 随机种子
    Returns:
        folds: 包含每个fold中文件路径列表的列表
    """
    # 设置随机种子确保fold划分的可复现性
    random.seed(random_seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)  # 打乱顺序

    # 计算每个fold的基本大小
    fold_size = len(files_shuffled) // num_folds
    remainder = len(files_shuffled) % num_folds
    
    folds = []
    start_idx = 0
    
    for i in range(num_folds):
        # 前remainder个fold多分配一个文件
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_fold_size
        folds.append(files_shuffled[start_idx:end_idx])
        start_idx = end_idx
    
    return folds


def get_train_test_files(folds, test_fold_index):
    """
    根据fold索引获取训练和测试文件列表
    Args:
        folds: fold划分列表
        test_fold_index: 测试fold的索引
    Returns:
        train_files: 训练文件路径列表
        test_files: 测试文件路径列表
    """
    test_files = folds[test_fold_index]
    train_files = [file for i, fold in enumerate(folds) if i != test_fold_index for file in fold]
    
    return train_files, test_files


def main(params, intra_mask_ratio, inter_mask_ratio):
    # 设置随机种子确保结果可复现
    set_seed(2025)
    
    # for record
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = 'board/' + current_time
    writer = SummaryWriter(log_dir)
    c_matrix = [[0., 0., 0., 0., 0.] for _ in range(5)]
    test_acc, test_mf1, test_kappa, test_mgm = [], [], [], []
    test_acc_per_class, test_recall_per_class, test_f1_per_class, test_gmean_per_class = ([0., 0., 0., 0., 0.],
                                                                                          [0., 0., 0., 0., 0.],
                                                                                          [0., 0., 0., 0., 0.],
                                                                                          [0., 0., 0., 0., 0.])

    lr = params['lr']
    weight_decay = params['weight_decay']
    total_epochs = params['total_epochs']
    warmup_epochs = params['warmup_epochs']
    min_lr = params['min_lr']
    now = time.gmtime()

    # start log
    logger = logging.getLogger("root")  # 创建独立的日志记录器
    logger.setLevel(logging.DEBUG)
    # 创建文件处理器
    file_handler = logging.FileHandler("log/train_MAE_{}.log".format(time.strftime("%m-%d-%H:%M:%S", now)))
    file_handler.setLevel(logging.DEBUG)
    # 创建格式器并添加到处理器
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)

    data_dir = 'dataset/SHHS/'  # 修改为SHHS数据集路径
    all_files = get_all_files(data_dir)
    folds = create_folds(all_files)

    for fold_num in range(len(folds)):
        train_files, test_files = get_train_test_files(folds, fold_num)
        # read train files
        array_data, array_label, transition_label = zip(
            *[load_train_dataset(file_name, num_epoches, 'PSD', 'label', step) for file_name in train_files])
        array_data = list(array_data)
        array_label = list(array_label)
        transition_label = list(transition_label)
        data_train = np.concatenate(array_data, axis=0)
        label_train = np.concatenate(array_label, axis=0)
        transition_train = np.concatenate(transition_label, axis=0)
        arr = transition_train.flatten()
        # 统计 0 和 1 的个数
        count_0 = np.sum(arr == 0)
        count_1 = np.sum(arr == 1)
        # 计算频率
        freq_0 = count_0 / len(arr)
        freq_1 = count_1 / len(arr)
        weights = torch.tensor([freq_0, freq_1]).float().to(device)

        # read test files
        array_data, array_label, transition_label = zip(
            *[load_dataset(file_name, num_epoches, 'PSD', 'label') for file_name in test_files])
        array_data = list(array_data)
        array_label = list(array_label)
        transition_label = list(transition_label)
        data_test = np.concatenate(array_data, axis=0)
        label_test = np.concatenate(array_label, axis=0)
        transition_test = np.concatenate(transition_label, axis=0)

        del array_data
        del array_label
        del transition_label

        # zscore normalization
        mean = np.mean(data_train)
        std = np.std(data_train)
        data_train = (data_train - mean) / std
        data_test = (data_test - mean) / std
        logger.info(
            "Fold {} has training data with shape {}, testing data with shape {}, mean value {} and std {}".format(
                fold_num, str(data_train.shape), str(data_test.shape), mean, std))
        # to tensor
        data_train, data_test = torch.tensor(data_train), torch.tensor(data_test)
        label_train, label_test = torch.tensor(label_train), torch.tensor(label_test)
        transition_train, transition_test = torch.tensor(transition_train), torch.tensor(transition_test)
        mean, std = torch.tensor(mean), torch.tensor(std)
        logging.info("Begin the fold {}".format(fold_num))

        # train classifier
        train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test, transition_train,
                                                transition_test, batch_size=128, seed=2025)
        model = MASS(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                         num_patches=num_patches, num_epoches=num_epoches, intra_mask_ratio=intra_mask_ratio,
                         inter_mask_ratio=inter_mask_ratio, intra_encoder_depth=intra_encoder_depth,
                         inter_encoder_depth=inter_encoder_depth, inter_hidden_dim=inter_hidden_dim, 
                         global_encoder_depth=global_encoder_depth, mean=mean,
                         std=std, seed=2025).to(device)
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        # 使用带有warmup的cosine学习率调度器
        # warmup_epochs为前10%的epoch，cosine annealing覆盖剩余90%的epoch
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps=total_epochs,
            warmup_steps=warmup_epochs,
            max_lr=lr,
            min_lr=min_lr
        )
        ep, model_best, acc, macro_f1, kappa, g_mean, acc_per_class, recall_per_class, f1_per_class, g_mean_per_class, cm = classifier_train(
            train_iter, test_iter, model, optimizer, scheduler, 'ft', total_epochs, fold_num, device, writer, weights)
        model_path = 'model/SHHS_intra_mask{}_enc{}_inter_mask{}_length{}_step{}/Classifier_ft_fold{}_acc{}.pth'.format(
            intra_mask_ratio, intra_encoder_depth, inter_mask_ratio, num_epoches, step, fold_num, acc)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model_best.state_dict(), model_path)

        for i in range(5):
            for j in range(5):
                c_matrix[i][j] += cm[i][j]
        test_acc.append(acc)
        test_mf1.append(macro_f1)
        test_kappa.append(kappa)
        test_mgm.append(g_mean)
        for i in range(5):
            test_acc_per_class[i] += acc_per_class[i]
            test_recall_per_class[i] += recall_per_class[i]
            test_f1_per_class[i] += f1_per_class[i]
            test_gmean_per_class[i] += g_mean_per_class[i]
        fold_num += 1
        logger.info('\n\n')
        # break

    mean_acc, std_acc = mean_std(test_acc)
    mean_mf1, std_mf1 = mean_std(test_mf1)
    mean_kappa, std_kappa = mean_std(test_kappa)
    mean_mgm, std_mgm = mean_std(test_mgm)
    logger.info('Average ACC is: {} and std is: {}'.format(mean_acc, std_acc))
    logger.info('Average MF1 is: {} and std is: {}'.format(mean_mf1, std_mf1))
    logger.info('Average Kappa is: {} and std is: {}'.format(mean_kappa, std_kappa))
    logger.info('Average MG-mean is: {} and std is: {}'.format(mean_mgm, std_mgm))
    for i in range(5):
        logger.info(
            'Class {} has average ACC is: {}'.format(i + 1, test_acc_per_class[i] / 20))
        logger.info(
            'Class {} has average RECALL is: {}'.format(i + 1, test_recall_per_class[i] / 20))
        logger.info(
            'Class {} has average F1 is: {}'.format(i + 1, test_f1_per_class[i] / 20))
        logger.info(
            'Class {} has average G-MEAN is: {}'.format(i + 1, test_gmean_per_class[i] / 20))
    logger.info('Confusion matrix is:\n{}'.format(c_matrix))

    logger.removeHandler(file_handler)


def parse_args():
    """
    解析命令行参数
    Returns:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='SleepMAE Training for SHHS Dataset')
    parser.add_argument('--intra_mask_ratio', type=float, default=0.5,
                        help='Intra-epoch mask ratio (default: 0.7)')
    parser.add_argument('--inter_mask_ratio', type=float, default=0.2,
                        help='Inter-epoch mask ratio (default: 0.4)')
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 设置全局随机种子
    set_seed(2025)
    
    # 调用主函数，传入mask ratio参数
    main(params, args.intra_mask_ratio, args.inter_mask_ratio)
    print("Finished!")
    os.system("shutdown -h 10")
