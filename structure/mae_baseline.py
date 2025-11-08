import torch
from torch import nn
import torch.nn.functional as F
from .transformer_layers import TransformerLayers
import math


class SleepMAEBaseline(nn.Module):
    """消融实验 - Baseline版本：无Mask无Prompt
    
    该版本移除了所有masking机制和prompt机制，直接对完整的输入序列进行建模。
    用于验证原始结构无辅助机制的表现。
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, num_patches, num_epoches, intra_mask_ratio,
                 inter_mask_ratio, global_encoder_depth, intra_encoder_depth, inter_encoder_depth, inter_hidden_dim, 
                 mean, std, use_cls_token=True, seed=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.intra_encoder_depth = intra_encoder_depth
        self.inter_encoder_depth = inter_encoder_depth
        self.mlp_ratio = mlp_ratio
        self.inter_hidden_dim = inter_hidden_dim
        self.mean = mean
        self.std = std
        # 测试时mask参数
        self.intra_mask_ratio = intra_mask_ratio
        self.inter_mask_ratio = inter_mask_ratio
        self.test_mask_enabled = False  # 控制测试时是否启用mask

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
        
        # intra-epoch Transformer encoder - 处理每个epoch内的patches
        # 输入：每个epoch的所有patches + epoch_cls_token
        # 输出：每个epoch的CLS token作为epoch级特征
        self.intra_encoder = TransformerLayers(embed_dim, intra_encoder_depth, mlp_ratio, num_heads, dropout)
        self.intra_encoder_norm = nn.LayerNorm(embed_dim)
        
        # inter-epoch Bi-GRU encoder（与完整MAE模型保持一致）
        # 输入：所有epochs的CLS tokens [B, total_epoches, embed_dim]
        # 输出：每个epoch的最终特征表示 [B, total_epoches, 2*inter_hidden_dim]
        self.inter_bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.inter_hidden_dim,
            num_layers=self.inter_encoder_depth,
            batch_first=True,
            bidirectional=True,  # 双向GRU，输出维度为2*hidden_dim
            dropout=dropout if self.inter_encoder_depth > 1 else 0
        )

        # 分类器 - 输入维度为2*inter_hidden_dim（与完整MAE模型保持一致）
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

    def enable_test_mask(self, enabled=True):
        """启用或禁用测试时的mask功能"""
        self.test_mask_enabled = enabled
    
    def apply_test_mask(self, eeg):
        """
        在测试时对输入数据应用mask，被mask的部分设置为0
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :return: masked_eeg [B, num_epoches, num_patches, 128]
        """
        if not self.test_mask_enabled:
            return eeg
            
        b, num_epoches, num_patches, feature_dim = eeg.shape
        masked_eeg = eeg.clone()
        
        # 1. Inter-epoch masking: 随机mask一些epochs
        if self.inter_mask_ratio > 0:
            num_masked_epochs = int(num_epoches * self.inter_mask_ratio)
            for batch_idx in range(b):
                # 为每个样本随机选择要mask的epochs
                masked_epoch_indices = torch.randperm(num_epoches)[:num_masked_epochs]
                # 将选中的epochs设置为0
                masked_eeg[batch_idx, masked_epoch_indices, :, :] = 0
        
        # 2. Intra-epoch masking: 在未被epoch-level mask的epochs中随机mask一些patches
        if self.intra_mask_ratio > 0:
            num_masked_patches = int(num_patches * self.intra_mask_ratio)
            for batch_idx in range(b):
                for epoch_idx in range(num_epoches):
                    # 只对未被epoch-level mask的epochs进行patch-level mask
                    if not torch.all(masked_eeg[batch_idx, epoch_idx] == 0):
                        # 为每个epoch随机选择要mask的patches
                        masked_patch_indices = torch.randperm(num_patches)[:num_masked_patches]
                        # 将选中的patches设置为0
                        masked_eeg[batch_idx, epoch_idx, masked_patch_indices, :] = 0
        
        return masked_eeg

    def create_sinusoidal_encoding(self, max_len, embed_dim):
        """创建正弦位置编码"""
        position = torch.arange(0, max_len).float().unsqueeze(1)  # shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # [embed_dim//2]
        sinusoidal_encoding = torch.zeros(max_len, embed_dim)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices
        return sinusoidal_encoding
    
    def intra_epoch_encoding(self, eeg):
        """
        intra-epoch建模：处理每个epoch内的patches
        无masking机制，使用所有patches
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :return: epoch_cls_tokens [B, num_epoches, embed_dim]
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 1. 批量进行patch embedding
        # 重塑为: [B * num_epoches, num_patches, 128]
        reshaped_patches = eeg.view(b * num_epoches, num_patches, 128)
        
        # Patch embedding: [B * num_epoches, num_patches, embed_dim]
        patch_tokens = self.patch_embedding(reshaped_patches)
        
        # 2. 构建输入序列：epoch_cls_token + patch_tokens
        # 扩展epoch_cls_token: [1, 1, embed_dim] -> [B * num_epoches, 1, embed_dim]
        epoch_cls_tokens_expanded = self.epoch_cls_token.expand(b * num_epoches, -1, -1)
        
        # 拼接CLS token和patch tokens: [B * num_epoches, 1 + num_patches, embed_dim]
        intra_input = torch.cat([epoch_cls_tokens_expanded, patch_tokens], dim=1)
        
        # 3. 添加位置编码
        # 为每个epoch内的序列构建位置编码
        full_pos_encoding = self.create_sinusoidal_encoding(num_patches + 1, self.embed_dim).to(eeg.device)  # +1 for CLS
        
        # 扩展位置编码到所有epochs: [B * num_epoches, 1 + num_patches, embed_dim]
        pos_encoding = full_pos_encoding.unsqueeze(0).expand(b * num_epoches, -1, -1)
        intra_input = intra_input + pos_encoding
        
        # 4. 批量Transformer编码（并行处理所有epoch）
        # 输入: [B * num_epoches, 1 + num_patches, embed_dim]
        # 输出: [B * num_epoches, 1 + num_patches, embed_dim]
        intra_hidden = self.intra_encoder(intra_input)
        intra_hidden = self.intra_encoder_norm(intra_hidden)
        
        # 5. 提取epoch CLS tokens
        # [B * num_epoches, embed_dim]
        epoch_cls_outputs = intra_hidden[:, 0, :]
        
        # 6. 重塑回原始batch维度: [B, num_epoches, embed_dim]
        epoch_cls_tokens = epoch_cls_outputs.view(b, num_epoches, self.embed_dim)
        
        return epoch_cls_tokens
    
    def inter_epoch_encoding(self, epoch_cls_tokens):
        """
        inter-epoch上下文学习：处理epochs间的时序关系
        使用Bi-GRU对epoch CLS tokens进行序列建模（与完整MAE模型保持一致）
        
        :param epoch_cls_tokens: epoch级特征 [B, num_epoches, embed_dim]
        :return: final_epoch_features [B, num_epoches, 2*inter_hidden_dim]
        """
        # 通过Bi-GRU进行序列建模: [B, num_epoches, embed_dim] -> [B, num_epoches, 2*inter_hidden_dim]
        # Bi-GRU能够捕获双向的时序依赖关系
        bigru_output, _ = self.inter_bigru(epoch_cls_tokens)
        
        return bigru_output

    def forward(self, eeg, temperature, transition=None):
        """
        Baseline版本前向传播：训练时无Mask无Prompt，测试时可选择性应用mask
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param temperature: 温度参数（保留兼容性）
        :param transition: transition标签 [B, num_epoches]
        :return: (sleep_stage_pred, transition_pred, transition_label)
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 在测试时应用mask（如果启用）
        # 训练时保持原有逻辑不变，测试时可选择性mask数据来验证鲁棒性
        if self.test_mask_enabled and not self.training:
            eeg = self.apply_test_mask(eeg)
        
        # 第一阶段：intra-epoch建模
        # 处理每个epoch内的patches，训练时无masking机制
        # epoch_cls_tokens: [B, num_epoches, embed_dim]
        epoch_cls_tokens = self.intra_epoch_encoding(eeg)
        
        # 第二阶段：inter-epoch上下文学习
        # 直接对epoch CLS tokens进行序列建模，无prompt机制
        # final_epoch_features: [B, num_epoches, embed_dim]
        final_epoch_features = self.inter_epoch_encoding(epoch_cls_tokens)
        
        # 分类任务
        # 1. 睡眠阶段分类：对所有epochs进行分类
        # final_epoch_features: [B, num_epoches, embed_dim] -> [B * num_epoches, embed_dim]
        all_epoch_features = final_epoch_features.contiguous().view(b * num_epoches, -1)
        # 睡眠阶段预测: [B * num_epoches, 5]
        sleep_stage_pred = self.sleep_stage_classifier(all_epoch_features)
        
        # 2. Transition分类：对所有epochs进行transition预测（无MAE任务）
        # transition预测: [B * num_epoches, 2]
        transition_pred = self.transition_classifier(all_epoch_features)
        
        # 如果提供了transition标签，重塑为对应维度
        if transition is not None:
            # [B, num_epoches] -> [B * num_epoches]
            transition_label = transition.contiguous().view(-1)
        else:
            transition_label = None
        
        return sleep_stage_pred, transition_pred, transition_label