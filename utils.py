from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
import random
import torch


class SleepDataset(Dataset):

    def __init__(self, x_tensor, y_tensor, z_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return len(self.x)


def load_dataloader(data_train, data_test, label1_train, label1_test, label2_train, label2_test, batch_size, seed=2025):
    # 设置随机数生成器确保数据加载的可复现性
    g = torch.Generator()
    g.manual_seed(seed)

    # 设置worker初始化函数，确保多进程数据加载的可复现性
    train_iter = DataLoader(dataset=SleepDataset(data_train, label1_train, label2_train), batch_size=batch_size,
                            shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True,
                            generator=g, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    test_iter = DataLoader(dataset=SleepDataset(data_test, label1_test, label2_test), batch_size=batch_size,
                           shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True,
                           worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    return train_iter, test_iter


# def generate_transition(label, mode):
#     # 获取label的形状
#     n, length = label.shape
#     # 创建一个和label形状相同的transition数组，初始值为0
#     transition = np.zeros_like(label, dtype=int)
#     # 根据不同的mode进行处理
#     if mode == 'left':
#         for i in range(n):
#             for j in range(length - 1):  # 遍历每个label[i,j]到label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # 标签不相等时
#                     transition[i, j] = 1
#     elif mode == 'right':
#         for i in range(n):
#             for j in range(length - 1):  # 遍历每个label[i,j]到label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # 标签不相等时
#                     transition[i, j + 1] = 1
#     elif mode == 'both':
#         for i in range(n):
#             for j in range(length - 1):  # 遍历每个label[i,j]到label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # 标签不相等时
#                     transition[i, j] = 1
#                     transition[i, j + 1] = 1
#     else:
#         raise ValueError("Mode must be one of 'left', 'right', or 'both'")
#     return transition

def generate_transition(label):
    n, length = label.shape
    transition = np.zeros((n, length), dtype=int)

    for i in range(n):
        for j in range(length):
            # 获取当前label[i, j]，以及边界情况处理
            current = label[i, j]
            prev = label[i, j - 1] if j > 0 else current
            next = label[i, j + 1] if j < length - 1 else current

            # 判断条件并赋值
            if current == prev and current == next:
                transition[i, j] = 0
            else:
                transition[i, j] = 1

    return transition


def load_dataset(load_path, length, data_key, label_key):
    filePath = load_path
    datasets = loadmat(filePath)
    dataAll = np.array(datasets[data_key], dtype=np.float32)
    labelAll = np.array(datasets[label_key], dtype=np.int64).flatten()
    # [N, 30, 128], [N]
    n, len, dim = dataAll.shape
    last_full_batch_end_idx = n - n % length
    if n % length != 0:
        # 计算需要从前面借用的数据量
        borrow_size = length - (n % length)
        # 创建一个新的数组，包含所有完整批次的数据和最后一个合并批次的数据
        dataAll = np.concatenate([dataAll[:last_full_batch_end_idx], np.concatenate(
            [dataAll[last_full_batch_end_idx - borrow_size:last_full_batch_end_idx],
             dataAll[last_full_batch_end_idx:]])]).reshape((n // length + 1), length, len, dim)
        labelAll = np.concatenate([labelAll[:last_full_batch_end_idx], np.concatenate(
            [labelAll[last_full_batch_end_idx - borrow_size:last_full_batch_end_idx],
             labelAll[last_full_batch_end_idx:]])]).reshape((n // length + 1), length)
    else:
        dataAll = dataAll.reshape(n // length, length, len, dim)
        labelAll = labelAll.reshape(n // length, length)
    transition_label = generate_transition(labelAll)
    return dataAll, labelAll, transition_label


def load_train_dataset(load_path, length, data_key, label_key, step):
    filePath = load_path
    datasets = loadmat(filePath)
    dataAll = np.array(datasets[data_key], dtype=np.float32)
    labelAll = np.array(datasets[label_key], dtype=np.int64).flatten()
    # [N, 30, 128], [N]
    n, len, dim = dataAll.shape
    start = 0
    data, label = [], []
    while start + length <= n:
        data.append(dataAll[start:start + length])
        label.append(labelAll[start:start + length])
        start += step
    data = np.array(data)
    label = np.array(label)
    transition_label = generate_transition(label)
    return data, label, transition_label


def load_lgsleepnet_dataset(load_path, data_key, label_key):
    filePath = load_path
    datasets = loadmat(filePath)
    data = np.array(datasets[data_key], dtype=np.float32)
    label = np.array(datasets[label_key], dtype=np.int64).flatten()
    return data, label


def mean_std(data):
    mean_acc = np.mean(data)
    std_acc = np.std(data)
    return mean_acc, std_acc

# file_dir = 'dataset/EDF20/mat/SC4001.mat'
# data, label, transition = load_dataset(file_dir, 32, 'PSD100', 'label')
