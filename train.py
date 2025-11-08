import numpy as np
from tqdm import tqdm, trange
import torch
import random
import logging
from sklearn.metrics import confusion_matrix, f1_score, recall_score, cohen_kappa_score, precision_score
import torch.nn.functional as F
import torch.nn as nn
from others.LGSleepNet import PolyLoss

def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
Poly = PolyLoss()
criterion = nn.CrossEntropyLoss()


def cal_mg_mean(y_true, y_pred, recall_per_class):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # 计算每个类别的特异性
    specificity_per_class = TN / (TN + FP)
    # 计算每个类别的
    g_mean_per_class = np.sqrt(specificity_per_class * recall_per_class)
    # 计算总的平均
    mg_mean_total = np.mean(g_mean_per_class)
    return mg_mean_total, g_mean_per_class


def classifier_test(test_iter, model, ep, temperature, device, logger=None):
    if logger is None:
        logger = logging.getLogger()
    logger.info("Begin testing on " + str(device) + "...")
    model.eval()
    with torch.no_grad():
        # all_labels = []
        # all_preds = []
        all_data = []  # 用于存储所有的 data
        all_labels = []  # 用于存储所有的 label
        all_transitions = []  # 用于存储所有的 transition
        for data, label, transition in test_iter:
            data = data.float().to(device)
            label = label.long().to(device).view(-1)
            transition = transition.long().to(device)
            all_data.append(data)
            all_labels.append(label)
            all_transitions.append(transition)
        # 将列表中的张量拼接为单个张量
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_transitions = torch.cat(all_transitions, dim=0)

        # 输入到神经网络并获取结果
        pred, _, _ = model(all_data, temperature, all_transitions)
        all_preds = pred.argmax(dim=1).tolist()  # 获取预测值
        all_labels = all_labels.tolist()  # 转为列表

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算指标
    acc = precision_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    # 计算每个类别的指标
    acc_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    g_mean, g_mean_per_class = cal_mg_mean(all_labels, all_preds, recall_per_class)
    # 输出log
    for i in range(5):
        logger.info(
            f'Class {i + 1} has Acc: {acc_per_class[i]}, Recall: {recall_per_class[i]}, F1: {f1_per_class[i]}, G-mean: {g_mean_per_class[i]}')
    logger.info(f'Test acc: {acc}, MF1: {macro_f1}, Kappa: {kappa}, MGm: {g_mean}')
    return acc, macro_f1, kappa, g_mean, acc_per_class, recall_per_class, f1_per_class, g_mean_per_class, cm


def classifier_train(train_iter, test_iter, model, optimizer, scheduler, stage, num_epochs, fold, device,
                     writer, weights, logger=None):
    if logger is None:
        logger = logging.getLogger()
    logger.info("Begin training classifier on " + str(device) + "...")
    acc_best, mf1_best, kappa_best, mgm_best = -1, -1, -1, -1
    acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = [], [], [], []
    cm_best = None
    ep_best = 0
    model_best = None
    if stage == 'lp':
        # linear probing
        logger.info('Starting linear probing')
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # fine-tuning
        logger.info('Starting fine-tuning')
        for param in model.parameters():
            param.requires_grad = True
    for ep in trange(num_epochs):
        batch_num = 0
        total_loss = 0.
        model.train()
        epoch_tensor = torch.tensor(ep, dtype=torch.float32)
        temperature = 1.0 * torch.exp(-0.005 * epoch_tensor)
        with tqdm(total=len(train_iter), desc=' Train on epoch: {}'.format(ep + 1), leave=True) as bar:
            for data, label, transition in train_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                transition = transition.long().to(device)
                # [B, 5]
                pred, transition_pred, transition_label = model(data, temperature, transition)
                loss1 = criterion(pred, label)
                one_hot_label = F.one_hot(label, num_classes=5)
                loss2 = 1 - F.cosine_similarity(pred, one_hot_label).mean()
                loss3 = F.cross_entropy(transition_pred, transition_label, weight=weights)
                loss = loss1 + 2 * loss2 + 0.5 * loss3
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                batch_num += 1
                bar.update(1)
        scheduler.step()
        logger.info('Epoch: {}, pred loss: {}'.format(ep + 1, total_loss / batch_num))
        acc_test, mf1_test, kappa_test, mgm_test, acc_per_class, recall_per_class, f1_per_class, gmean_per_class, cm = classifier_test(
            test_iter, model, ep, temperature, device=device, logger=logger)
        writer.add_scalars(main_tag='Test loss in classifier with fold {}'.format(fold),
                           tag_scalar_dict={'test pred acc': acc_test,
                                            'test MF1': mf1_test,
                                            'test Kappa': kappa_test,
                                            'test MGmean': mgm_test,
                                            'pred mse loss': total_loss / batch_num},
                           global_step=int(ep + 1))
        if acc_test > acc_best:
            acc_best, mf1_best, kappa_best, mgm_best = acc_test, mf1_test, kappa_test, mgm_test
            acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = acc_per_class, recall_per_class, f1_per_class, gmean_per_class
            cm_best = cm
            ep_best = ep + 1
            model_best = model
        logger.info('Best test acc: {} in epoch: {} '.format(acc_best, ep_best))
        if (ep + 1) % 50 == 0:
            model_path = 'checkpoint/Classifier_{}_epoch{}_acc{}_random{}.pth'.format(stage, ep_best, acc_best,
                                                                                      random.randint(1, 1000))
            torch.save(model_best.state_dict(), model_path)
    logger.info('\n')
    return ep_best, model_best, acc_best, mf1_best, kappa_best, mgm_best, acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best, cm_best


def ablation_test(test_iter, model, ep, temperature, device, enable_test_mask=False, logger=None):
    """
    消融模型测试函数，支持测试时mask功能
    
    :param test_iter: 测试数据迭代器
    :param model: 消融模型（Baseline/MaskOnly/PromptOnly）
    :param ep: 当前epoch
    :param temperature: 温度参数
    :param device: 设备
    :param enable_test_mask: 是否启用测试时mask
    :param logger: 日志记录器
    :return: 测试指标
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info(f"Begin testing ablation model on {device} (test_mask={enable_test_mask})...")
    model.eval()
    
    # 设置测试时mask状态
    if hasattr(model, 'enable_test_mask'):
        model.enable_test_mask(enable_test_mask)
    
    with torch.no_grad():
        all_data = []  # 用于存储所有的 data
        all_labels = []  # 用于存储所有的 label
        all_transitions = []  # 用于存储所有的 transition
        for data, label, transition in test_iter:
            data = data.float().to(device)
            label = label.long().to(device).view(-1)
            transition = transition.long().to(device)
            all_data.append(data)
            all_labels.append(label)
            all_transitions.append(transition)
        # 将列表中的张量拼接为单个张量
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_transitions = torch.cat(all_transitions, dim=0)

        # 输入到神经网络并获取结果
        pred, _, _ = model(all_data, temperature, all_transitions)
        all_preds = pred.argmax(dim=1).tolist()  # 获取预测值
        all_labels = all_labels.tolist()  # 转为列表

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算指标
    acc = precision_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    # 计算每个类别的指标
    acc_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    g_mean, g_mean_per_class = cal_mg_mean(all_labels, all_preds, recall_per_class)
    # 输出log
    for i in range(5):
        logger.info(
            f'Class {i + 1} has Acc: {acc_per_class[i]}, Recall: {recall_per_class[i]}, F1: {f1_per_class[i]}, G-mean: {g_mean_per_class[i]}')
    logger.info(f'Test acc: {acc}, MF1: {macro_f1}, Kappa: {kappa}, MGm: {g_mean}')
    return acc, macro_f1, kappa, g_mean, acc_per_class, recall_per_class, f1_per_class, g_mean_per_class, cm


def ablation_train(train_iter, test_iter, model, optimizer, scheduler, stage, num_epochs, fold, device,
                   writer, weights, model_name="ablation", logger=None):
    """
    消融模型训练函数，支持线性探测和微调两个阶段
    
    :param train_iter: 训练数据迭代器
    :param test_iter: 测试数据迭代器
    :param model: 消融模型（Baseline/MaskOnly/PromptOnly）
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param stage: 训练阶段（'lp'为线性探测，'ft'为微调）
    :param num_epochs: 训练轮数
    :param fold: 折数
    :param device: 设备
    :param writer: TensorBoard writer
    :param weights: 类别权重
    :param model_name: 模型名称（用于保存）
    :param logger: 日志记录器
    :return: 训练结果
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info(f"Begin training {model_name} ablation model on {device}...")
    acc_best, mf1_best, kappa_best, mgm_best = -1, -1, -1, -1
    acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = [], [], [], []
    cm_best = None
    ep_best = 0
    model_best = None
    
    # 确保测试时mask功能在训练期间关闭
    if hasattr(model, 'enable_test_mask'):
        model.enable_test_mask(False)
    
    if stage == 'lp':
        # linear probing
        logger.info('Starting linear probing')
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # fine-tuning
        logger.info('Starting fine-tuning')
        for param in model.parameters():
            param.requires_grad = True
    
    for ep in trange(num_epochs):
        batch_num = 0
        total_loss = 0.
        model.train()
        epoch_tensor = torch.tensor(ep, dtype=torch.float32)
        temperature = 1.0 * torch.exp(-0.005 * epoch_tensor)
        with tqdm(total=len(train_iter), desc=f' Train {model_name} on epoch: {ep + 1}', leave=True) as bar:
            for data, label, transition in train_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                transition = transition.long().to(device)
                # [B, 5]
                pred, transition_pred, transition_label = model(data, temperature, transition)
                loss1 = criterion(pred, label)
                one_hot_label = F.one_hot(label, num_classes=5)
                loss2 = 1 - F.cosine_similarity(pred, one_hot_label).mean()
                loss3 = F.cross_entropy(transition_pred, transition_label, weight=weights)
                loss = loss1 + 2 * loss2 + 0.5 * loss3
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                batch_num += 1
                bar.update(1)
        scheduler.step()
        logger.info(f'Epoch: {ep + 1}, pred loss: {total_loss / batch_num}')
        
        # 测试
        acc_test, mf1_test, kappa_test, mgm_test, acc_per_class, recall_per_class, f1_per_class, gmean_per_class, cm = ablation_test(
            test_iter, model, ep, temperature, device=device, enable_test_mask=True, logger=logger)
        
        writer.add_scalars(main_tag=f'Test loss in {model_name} ablation with fold {fold}',
                           tag_scalar_dict={'test pred acc': acc_test,
                                            'test MF1': mf1_test,
                                            'test Kappa': kappa_test,
                                            'test MGmean': mgm_test,
                                            'pred mse loss': total_loss / batch_num},
                           global_step=int(ep + 1))
        if acc_test > acc_best:
            acc_best, mf1_best, kappa_best, mgm_best = acc_test, mf1_test, kappa_test, mgm_test
            acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = acc_per_class, recall_per_class, f1_per_class, gmean_per_class
            cm_best = cm
            ep_best = ep + 1
            model_best = model
        logger.info(f'Best test acc: {acc_best} in epoch: {ep_best}')
        if (ep + 1) % 50 == 0:
            model_path = f'checkpoint/{model_name}_Ablation_{stage}_epoch{ep_best}_acc{acc_best}_random{random.randint(1, 1000)}.pth'
            torch.save(model_best.state_dict(), model_path)
    logger.info('\n')
    return ep_best, model_best, acc_best, mf1_best, kappa_best, mgm_best, acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best, cm_best


def lgsleepnet_test(test_iter, model, ep, device):
    logging.info("Begin testing LGSleepNet on " + str(device) + "...")
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        with tqdm(total=len(test_iter), desc=' Test on epoch: {}'.format(ep + 1), leave=True) as bar:
            for data, label, _ in test_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                # [B, 5]
                pred = model(data)
                all_preds.extend(pred.argmax(dim=1).tolist())
                all_labels.extend(label.tolist())
                bar.update(1)
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算指标
    acc = precision_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    # 计算每个类别的指标
    acc_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    g_mean, g_mean_per_class = cal_mg_mean(all_labels, all_preds, recall_per_class)
    # 输出log
    for i in range(5):
        logging.info(
            f'Class {i + 1} has Acc: {acc_per_class[i]}, Recall: {recall_per_class[i]}, F1: {f1_per_class[i]}, G-mean: {g_mean_per_class[i]}')
    logging.info(f'Test acc: {acc}, MF1: {macro_f1}, Kappa: {kappa}, MGm: {g_mean}')
    return acc, macro_f1, kappa, g_mean, acc_per_class, recall_per_class, f1_per_class, g_mean_per_class, cm


def lgsleepnet_train(train_iter, test_iter, model, optimizer, num_epochs, fold, device, writer):
    logging.info("Begin training LGSleepNet on " + str(device) + "...")
    acc_best, mf1_best, kappa_best, mgm_best = -1, -1, -1, -1
    acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = [], [], [], []
    cm_best = None
    ep_best = 0
    model_best = None
    for ep in trange(num_epochs):
        batch_num = 0
        total_loss = 0.
        model.train()
        with tqdm(total=len(train_iter), desc=' Train on epoch: {}'.format(ep + 1), leave=True) as bar:
            for data, label, _ in train_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                # [B, 5]
                pred = model(data)
                loss = Poly(pred, label)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_num += 1
                bar.update(1)
        logging.info('Epoch: {}, pred loss: {}'.format(ep + 1, total_loss / batch_num))
        acc_test, mf1_test, kappa_test, mgm_test, acc_per_class, recall_per_class, f1_per_class, gmean_per_class, cm = lgsleepnet_test(
            test_iter, model, ep, device=device)
        writer.add_scalars(main_tag='Test loss in classifier with fold {}'.format(fold),
                           tag_scalar_dict={'test pred acc': acc_test,
                                            'test MF1': mf1_test,
                                            'test Kappa': kappa_test,
                                            'test MGmean': mgm_test,
                                            'pred mse loss': total_loss / batch_num},
                           global_step=int(ep + 1))
        if acc_test > acc_best:
            acc_best, mf1_best, kappa_best, mgm_best = acc_test, mf1_test, kappa_test, mgm_test
            acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = acc_per_class, recall_per_class, f1_per_class, gmean_per_class
            cm_best = cm
            ep_best = ep + 1
            model_best = model
        logging.info('Best test acc: {} in epoch: {} '.format(acc_best, ep_best))
        if (ep + 1) % 20 == 0:
            model_path = 'checkpoint/LGSleepNet_epoch{}_acc{}_random{}.pth'.format(ep_best, acc_best,
                                                                                   random.randint(1, 1000))
            torch.save(model_best.state_dict(), model_path)
        elif ep + 1 - ep_best > 30:
            # early exit
            break
    logging.info('\n')
    return ep_best, model_best, acc_best, mf1_best, kappa_best, mgm_best, acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best, cm_best


def test(test_iter, model, ep, device):
    logging.info("Begin testing on " + str(device) + "...")
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        with tqdm(total=len(test_iter), desc=' Test on epoch: {}'.format(ep + 1), leave=True) as bar:
            for data, label, _ in test_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                # [B, 5]
                pred = model(data)
                all_preds.extend(pred.argmax(dim=1).tolist())
                all_labels.extend(label.tolist())
                bar.update(1)
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算指标
    acc = precision_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    # 计算每个类别的指标
    acc_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    g_mean, g_mean_per_class = cal_mg_mean(all_labels, all_preds, recall_per_class)
    # 输出log
    for i in range(5):
        logging.info(
            f'Class {i + 1} has Acc: {acc_per_class[i]}, Recall: {recall_per_class[i]}, F1: {f1_per_class[i]}, G-mean: {g_mean_per_class[i]}')
    logging.info(f'Test acc: {acc}, MF1: {macro_f1}, Kappa: {kappa}, MGm: {g_mean}')
    return acc, macro_f1, kappa, g_mean, acc_per_class, recall_per_class, f1_per_class, g_mean_per_class, cm


def train(train_iter, test_iter, model, optimizer, scheduler, num_epochs, fold, device, writer):
    logging.info("Begin training on " + str(device) + "...")
    acc_best, mf1_best, kappa_best, mgm_best = -1, -1, -1, -1
    acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = [], [], [], []
    cm_best = None
    ep_best = 0
    model_best = None
    for ep in trange(num_epochs):
        batch_num = 0
        total_loss = 0.
        model.train()
        with tqdm(total=len(train_iter), desc=' Train on epoch: {}'.format(ep + 1), leave=True) as bar:
            for data, label, _ in train_iter:
                data = data.float().to(device)
                label = label.long().to(device).view(-1)
                # [B, 5]
                pred = model(data)
                loss = criterion(pred, label)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_num += 1
                bar.update(1)
        scheduler.step()
        logging.info('Epoch: {}, pred loss: {}'.format(ep + 1, total_loss / batch_num))
        acc_test, mf1_test, kappa_test, mgm_test, acc_per_class, recall_per_class, f1_per_class, gmean_per_class, cm = lgsleepnet_test(
            test_iter, model, ep, device=device)
        writer.add_scalars(main_tag='Test loss in classifier with fold {}'.format(fold),
                           tag_scalar_dict={'test pred acc': acc_test,
                                            'test MF1': mf1_test,
                                            'test Kappa': kappa_test,
                                            'test MGmean': mgm_test,
                                            'pred mse loss': total_loss / batch_num},
                           global_step=int(ep + 1))
        if acc_test > acc_best:
            acc_best, mf1_best, kappa_best, mgm_best = acc_test, mf1_test, kappa_test, mgm_test
            acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best = acc_per_class, recall_per_class, f1_per_class, gmean_per_class
            cm_best = cm
            ep_best = ep + 1
            model_best = model
        logging.info('Best test acc: {} in epoch: {} '.format(acc_best, ep_best))
        if (ep + 1) % 20 == 0:
            model_path = 'checkpoint/MaskSleepNet_epoch{}_acc{}_random{}.pth'.format(ep_best, acc_best,
                                                                                     random.randint(1, 1000))
            torch.save(model_best.state_dict(), model_path)
        elif ep + 1 - ep_best > 25 and ep > 100:
            # early exit
            break
    logging.info('\n')
    return ep_best, model_best, acc_best, mf1_best, kappa_best, mgm_best, acc_per_class_best, recall_per_class_best, f1_per_class_best, gmean_per_class_best, cm_best
