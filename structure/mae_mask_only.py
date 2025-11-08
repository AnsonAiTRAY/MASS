import torch
from torch import nn
import torch.nn.functional as F
from .mask import MaskGenerator
from .transformer_layers import TransformerLayers
import math


class SleepMAEMaskOnly(nn.Module):
    """消融实验 - Mask Only版本：仅使用Mask机制，无Prompt
    
    该版本保留了masking机制但移除了prompt机制，用于验证遮蔽策略是否有效。
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, num_patches, num_epoches, intra_mask_ratio,
                 inter_mask_ratio, intra_encoder_depth, inter_encoder_depth, inter_hidden_dim, 
                 mean, std, use_cls_token=True, seed=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.intra_mask_ratio = intra_mask_ratio
        self.inter_mask_ratio = inter_mask_ratio
        self.intra_encoder_depth = intra_encoder_depth
        self.inter_encoder_depth = inter_encoder_depth
        self.mlp_ratio = mlp_ratio
        self.inter_hidden_dim = inter_hidden_dim
        self.mean = mean
        self.std = std
        # 测试时mask参数
        self.test_mask_enabled = False  # 控制测试时是否启用额外mask

        # patch embedding层 - 将原始patch特征[128]转换为embed_dim维度的token
        # 输入: [B, num_epoches, num_patches, 128]
        # 输出: [B, num_epoches, num_patches, embed_dim]
        self.patch_embedding = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )
        
        # epoch级CLS token - 用于每个epoch内的特征聚合
        # 维度: [1, embed_dim]，在每个epoch的patches前添加
        self.epoch_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # masking generators - 保留masking机制
        self.intra_mask = MaskGenerator(num_patches, intra_mask_ratio, seed=seed)
        self.inter_mask = MaskGenerator(num_epoches, inter_mask_ratio, seed=seed)
        
        # intra-epoch Transformer encoder - 处理每个epoch内的可见patches
        # 输入：每个epoch的可见patches + epoch_cls_token（无prompt）
        # 输出：每个epoch的CLS token作为epoch级特征
        self.intra_encoder = TransformerLayers(embed_dim, intra_encoder_depth, mlp_ratio, num_heads, dropout)
        self.intra_encoder_norm = nn.LayerNorm(embed_dim)
        
        # inter-epoch Bi-GRU encoder - 处理epoch间的时序关系
        # 输入：可见epochs的CLS tokens [B, num_visible_epoches, embed_dim]
        # 输出：每个epoch的最终特征表示 [B, total_epoches, 2*inter_hidden_dim]
        self.inter_bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.inter_hidden_dim,
            num_layers=self.inter_encoder_depth,
            batch_first=True,
            bidirectional=True,  # 双向GRU，输出维度为2*hidden_dim
            dropout=dropout if self.inter_encoder_depth > 1 else 0
        )

        # 分类器 - 输入维度为2*inter_hidden_dim（Bi-GRU输出维度）
        self.transition_classifier = nn.Sequential(
            nn.Linear(2*self.inter_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.sleep_stage_classifier = nn.Sequential(
            nn.Linear(2*self.inter_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(param, nn.Conv1d) or isinstance(param, nn.Conv2d):
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif isinstance(param, nn.Linear):
                    nn.init.xavier_uniform_(param)
                elif isinstance(param, nn.LayerNorm):
                    nn.init.ones_(param.bias)
                    nn.init.zeros_(param.weight)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def create_sinusoidal_encoding(self, max_len, embed_dim):
        """创建正弦位置编码"""
        position = torch.arange(0, max_len).float().unsqueeze(1)  # shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # [embed_dim//2]
        sinusoidal_encoding = torch.zeros(max_len, embed_dim)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices
        return sinusoidal_encoding
    
    def intra_epoch_encoding(self, eeg, unmasked_epoches_index_batch, patch_masks_dict):
        """
        intra-epoch建模：基于masking处理每个epoch内的可见patches（向量化优化版本）
        无prompt机制，只使用epoch CLS token
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param unmasked_epoches_index_batch: 可见epochs索引 [B, num_visible_epoches]
        :param patch_masks_dict: 每个样本每个epoch的patch mask信息
        :return: epoch_cls_tokens [B, num_visible_epoches, embed_dim]
        """
        b, _, num_patches, _ = eeg.shape
        num_visible_epoches = unmasked_epoches_index_batch.shape[1]

        # 1. 批量提取所有可见epoch的数据（向量化操作）
        # 使用高级索引直接提取所有可见epochs的数据
        # eeg: [B, E, P, D] -> visible_epoches_data: [B, num_visible_epoches, P, D]
        batch_indices = torch.arange(b).view(-1, 1).to(eeg.device)
        visible_epoches_data = eeg[batch_indices, unmasked_epoches_index_batch]

        # 2. 从 patch_masks_dict 恢复 unmasked_patches_idx_batch（向量化操作）
        # 构建批量patch索引张量: [B, num_visible_epoches, num_visible_patches]
        unmasked_patches_idx_batch = torch.stack(
            [torch.stack([patch_masks_dict[i][epoch_idx.item()] for epoch_idx in unmasked_epoches_index_batch[i]]) for i in range(b)]
        ).to(eeg.device)

        # 3. 批量提取可见patches（向量化操作）
        # visible_epoches_data: [B, num_visible_epoches, P, D]
        # unmasked_patches_idx_batch: [B, num_visible_epoches, num_visible_patches]
        # visible_patches: [B, num_visible_epoches, num_visible_patches, D]
        batch_indices = torch.arange(b).view(b, 1, 1).to(eeg.device)
        epoch_indices = torch.arange(num_visible_epoches).view(1, num_visible_epoches, 1).to(eeg.device)
        visible_patches = visible_epoches_data[batch_indices, epoch_indices, unmasked_patches_idx_batch]

        # 4. 批量进行patch embedding
        # 重塑为: [B * num_visible_epoches, num_visible_patches, 128]
        reshaped_patches = visible_patches.view(b * num_visible_epoches, -1, visible_patches.shape[-1])
        
        # Patch embedding: [B * num_visible_epoches, num_visible_patches, embed_dim]
        patch_tokens = self.patch_embedding(reshaped_patches)
        
        # 5. 构建输入序列：epoch_cls_token + patch_tokens（无prompt）
        # 扩展epoch_cls_token: [1, 1, embed_dim] -> [B * num_visible_epoches, 1, embed_dim]
        epoch_cls_tokens_expanded = self.epoch_cls_token.expand(b * num_visible_epoches, -1, -1)
        
        # 拼接CLS token和patch tokens: [B * num_visible_epoches, 1 + num_visible_patches, embed_dim]
        intra_input = torch.cat([epoch_cls_tokens_expanded, patch_tokens], dim=1)
        
        # 6. 添加位置编码（向量化优化版本）
        max_patches = num_patches
        full_pos_encoding = self.create_sinusoidal_encoding(max_patches + 1, self.embed_dim).to(eeg.device)  # +1 for CLS
        
        # CLS token位置编码（所有epoch的CLS都使用位置0）
        cls_pos = full_pos_encoding[0:1].expand(b * num_visible_epoches, -1, -1)
        
        # 可见patches位置编码（批量索引）
        # unmasked_patches_idx_batch: [B, num_visible_epoches, num_visible_patches]
        # 重塑为: [B * num_visible_epoches, num_visible_patches]
        patch_pos_indices = unmasked_patches_idx_batch.view(b * num_visible_epoches, -1)
        # 为patches添加偏移+1（因为CLS占用位置0）
        patch_pos = full_pos_encoding[patch_pos_indices + 1]
        
        # 拼接位置编码: [B * num_visible_epoches, 1 + num_visible_patches, embed_dim]
        pos_encoding = torch.cat([cls_pos, patch_pos], dim=1)
        intra_input = intra_input + pos_encoding
        
        # 7. 批量Transformer编码（并行处理所有epoch）
        # 输入: [B * num_visible_epoches, 1 + num_visible_patches, embed_dim]
        # 输出: [B * num_visible_epoches, 1 + num_visible_patches, embed_dim]
        intra_hidden = self.intra_encoder(intra_input)
        intra_hidden = self.intra_encoder_norm(intra_hidden)
        
        # 8. 提取epoch CLS tokens
        # [B * num_visible_epoches, embed_dim]
        epoch_cls_outputs = intra_hidden[:, 0, :]
        
        # 9. 重塑回原始batch维度: [B, num_visible_epoches, embed_dim]
        epoch_cls_tokens = epoch_cls_outputs.view(b, num_visible_epoches, self.embed_dim)
        
        return epoch_cls_tokens
    
    def inter_epoch_encoding(self, epoch_cls_tokens, unmasked_epoches_index_batch, masked_epoches_index_batch):
        """
        inter-epoch上下文学习：处理epochs间的时序关系（向量化优化版本）
        使用Bi-GRU进行序列建模，无prompt机制
        
        :param epoch_cls_tokens: epoch级特征 [B, num_visible_epoches, embed_dim]
        :param unmasked_epoches_index_batch: 可见epochs索引
        :param masked_epoches_index_batch: 遮蔽epochs索引
        :return: final_epoch_features [B, total_epoches, 2*inter_hidden_dim]
        """
        b, num_visible_epoches, _ = epoch_cls_tokens.shape
        num_masked_epoches = masked_epoches_index_batch.shape[1]
        total_epoches = num_visible_epoches + num_masked_epoches
        
        # 创建完整的epoch序列（向量化操作）
        # 初始化全零张量: [B, total_epoches, embed_dim]
        full_epoch_sequences = torch.zeros(b, total_epoches, self.embed_dim, device=epoch_cls_tokens.device)
        
        # 批量填入可见epochs的特征（向量化操作）
        # 使用高级索引一次性填入所有样本的可见epoch特征
        batch_indices = torch.arange(b).view(-1, 1).to(epoch_cls_tokens.device)
        full_epoch_sequences[batch_indices, unmasked_epoches_index_batch] = epoch_cls_tokens
        
        # 使用Bi-GRU进行inter-epoch建模（无prompt拼接）
        # 输入: [B, total_epoches, embed_dim]
        # 输出: [B, total_epoches, 2*inter_hidden_dim]
        bigru_output, _ = self.inter_bigru(full_epoch_sequences)
        
        return bigru_output

    def forward(self, eeg, temperature, transition=None):
        """
        Mask Only版本前向传播：使用Mask机制但无Prompt
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param temperature: 温度参数（保留兼容性）
        :param transition: transition标签 [B, num_epoches]
        :return: (sleep_stage_pred, transition_pred, transition_label)
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 第一阶段：生成masking信息（无全局prompt生成，向量化优化版本）
        # 生成epoch级mask
        unmasked_epoches_index_batch, masked_epoches_index_batch = self.inter_mask(batch_size=b)
        num_visible_epoches = unmasked_epoches_index_batch.shape[1]
        
        # 批量生成所有可见epoch的patch级mask（向量化操作）
        # 为所有样本的所有可见epoch一次性生成patch mask
        unmasked_patches_idx_batch, _ = self.intra_mask(batch_size=b * num_visible_epoches)
        unmasked_patches_idx_batch = unmasked_patches_idx_batch.view(b, num_visible_epoches, -1)
        
        # 构建patch_masks_dict（为了兼容后续函数）
        patch_masks_dict = {}
        for i in range(b):
            sample_masks = {}
            for j, epoch_idx in enumerate(unmasked_epoches_index_batch[i]):
                sample_masks[epoch_idx.item()] = unmasked_patches_idx_batch[i, j]
            patch_masks_dict[i] = sample_masks
        
        # 第二阶段：intra-epoch建模
        # 处理每个可见epoch内的可见patches，无prompt机制
        # epoch_cls_tokens: [B, num_visible_epoches, embed_dim]
        epoch_cls_tokens = self.intra_epoch_encoding(eeg, unmasked_epoches_index_batch, patch_masks_dict)
        
        # 第三阶段：inter-epoch上下文学习
        # 对epoch CLS tokens进行序列建模，无prompt机制
        # final_epoch_features: [B, total_epoches, embed_dim]
        final_epoch_features = self.inter_epoch_encoding(epoch_cls_tokens, unmasked_epoches_index_batch, masked_epoches_index_batch)
        
        # 分类任务
        # 1. 睡眠阶段分类：对所有epochs进行分类
        # final_epoch_features: [B, total_epoches, embed_dim] -> [B * total_epoches, embed_dim]
        all_epoch_features = final_epoch_features.contiguous().view(b * num_epoches, -1)
        # 睡眠阶段预测: [B * total_epoches, 5]
        sleep_stage_pred = self.sleep_stage_classifier(all_epoch_features)
        
        # 2. Transition分类：只对遮蔽的epochs进行transition预测（MAE任务）
        transition_pred_list = []
        transition_label_list = []
        
        for i in range(b):
            # 获取第i个样本的遮蔽epochs索引
            sample_masked_idx = masked_epoches_index_batch[i]
            
            if len(sample_masked_idx) > 0:  # 确保有遮蔽的epochs
                # 获取遮蔽epochs的特征: [num_masked_epoches, embed_dim]
                sample_masked_features = final_epoch_features[i, sample_masked_idx, :]
                # transition预测: [num_masked_epoches, 2]
                sample_transition_pred = self.transition_classifier(sample_masked_features)
                transition_pred_list.append(sample_transition_pred)
                
                # 获取对应的transition标签: [num_masked_epoches]
                if transition is not None:
                    sample_transition_label = transition[i, sample_masked_idx]
                    transition_label_list.append(sample_transition_label)
        
        # 拼接所有样本的transition预测结果
        if len(transition_pred_list) > 0:
            # [total_masked_epoches, 2]
            transition_pred = torch.cat(transition_pred_list, dim=0)
            if transition is not None and len(transition_label_list) > 0:
                # [total_masked_epoches]
                transition_label = torch.cat(transition_label_list, dim=0)
            else:
                transition_label = None
        else:
            # 如果没有遮蔽的epochs，创建空的tensor
            transition_pred = torch.empty(0, 2, device=eeg.device)
            transition_label = torch.empty(0, device=eeg.device) if transition is not None else None
        
        return sleep_stage_pred, transition_pred, transition_label