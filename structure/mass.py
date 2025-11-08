import torch
from torch import nn
import torch.nn.functional as F
from .mask import MaskGenerator
from .transformer_layers import TransformerLayers
import math


class MASS(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, num_patches, num_epoches, intra_mask_ratio,
                 inter_mask_ratio, global_encoder_depth, intra_encoder_depth, inter_encoder_depth, inter_hidden_dim, 
                 mean, std, use_cls_token=True, seed=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.intra_mask_ratio = intra_mask_ratio
        self.inter_mask_ratio = inter_mask_ratio
        self.global_encoder_depth = global_encoder_depth
        self.intra_encoder_depth = intra_encoder_depth
        self.inter_encoder_depth = inter_encoder_depth
        self.mlp_ratio = mlp_ratio
        self.inter_hidden_dim = inter_hidden_dim
        self.mean = mean
        self.std = std

        # 第一阶段：patch embedding层
        # 将原始patch特征[128]转换为embed_dim维度的token
        self.patch_embedding = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )
        
        # 全局CLS token - 用于生成全局prompt token
        # 维度: [1, embed_dim]，在所有可见patches前添加
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # epoch级CLS token - 用于每个epoch内的特征聚合
        # 维度: [1, embed_dim]，在每个epoch的patches前添加
        self.epoch_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # masking generators
        self.intra_mask = MaskGenerator(num_patches, intra_mask_ratio, seed=seed)
        self.inter_mask = MaskGenerator(num_epoches, inter_mask_ratio, seed=seed)

        # 第一阶段：全局Transformer encoder - 用于生成prompt token
        # 输入：所有可见patches + global_cls_token
        # 输出：global prompt token
        self.global_encoder = TransformerLayers(embed_dim, global_encoder_depth, mlp_ratio, num_heads, dropout)
        self.global_encoder_norm = nn.LayerNorm(embed_dim)
        
        # 第二阶段：intra-epoch Transformer encoder
        # 输入：每个epoch的可见patches + prompt_token + epoch_cls_token
        # 输出：每个epoch的CLS token作为epoch级特征
        self.intra_encoder = TransformerLayers(embed_dim, intra_encoder_depth, mlp_ratio, num_heads, dropout)
        self.intra_encoder_norm = nn.LayerNorm(embed_dim)
        
        # 第三阶段：inter-epoch Bi-GRU encoder
        # 输入：拼接后的特征 [B, total_epoches, 2*embed_dim] (prompt_token + epoch_cls_token)
        # 输出：每个epoch的最终特征表示 [B, total_epoches, 2*embed_dim] (保持双向GRU的输出维度)
        self.inter_bigru = nn.GRU(input_size=2*embed_dim, hidden_size=self.inter_hidden_dim, 
                                  num_layers=self.inter_encoder_depth, batch_first=True, bidirectional=True, dropout=dropout)
        # 直接使用2*embed_dim维度，无需投影层
        self.inter_norm = nn.LayerNorm(2*self.inter_hidden_dim)

        # 分类器 - 输入维度为2*embed_dim
        self.transition_classifier = nn.Sequential(
            nn.Linear(2*self.inter_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.sleep_stage_classifier = nn.Sequential(
            nn.Linear(2*self.inter_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        # positional encoding
        # nn.init.uniform_(self.intra_position_embedding, -.02, .02)
        # nn.init.uniform_(self.inter_position_embedding, -.02, .02)

    def create_sinusoidal_encoding(self, max_len, embed_dim):
        position = torch.arange(0, max_len).float().unsqueeze(1)  # shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # [embed_dim//2]
        sinusoidal_encoding = torch.zeros(max_len, embed_dim)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices
        return sinusoidal_encoding

    def generate_global_prompt(self, eeg):
        """
        第一阶段：生成全局prompt token (向量化版本)
        从所有可见patches中学习全局上下文信息

        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :return: prompt_token [B, embed_dim] - 全局prompt token
                unmasked_epoches_index_batch - 可见epochs索引
                masked_epoches_index_batch - 遮蔽epochs索引
                patch_masks_dict - 每个可见epoch的patch mask信息
        """
        b, num_epoches, num_patches, _ = eeg.shape

        # 1. 确定epoch和patch的遮蔽情况
        unmasked_epoches_index_batch, masked_epoches_index_batch = self.inter_mask(batch_size=b)
        num_visible_epoches = unmasked_epoches_index_batch.shape[1]

        # 为每个可见epoch生成patch级mask
        unmasked_patches_idx_batch, _ = self.intra_mask(batch_size=b * num_visible_epoches)
        unmasked_patches_idx_batch = unmasked_patches_idx_batch.view(b, num_visible_epoches, -1)

        # 2. 收集所有可见patches
        # 使用高级索引直接提取所有可见epochs的数据
        # eeg: [B, E, P, D] -> visible_epoches_data: [B, num_visible_epoches, P, D]
        batch_indices = torch.arange(b).view(-1, 1).to(eeg.device)
        visible_epoches_data = eeg[batch_indices, unmasked_epoches_index_batch]

        # 提取可见patches
        # visible_epoches_data: [B, num_visible_epoches, P, D]
        # unmasked_patches_idx_batch: [B, num_visible_epoches, num_visible_patches]
        # all_patches: [B, num_visible_epoches, num_visible_patches, D]
        batch_indices = torch.arange(b).view(b, 1, 1).to(eeg.device)
        epoch_indices = torch.arange(num_visible_epoches).view(1, num_visible_epoches, 1).to(eeg.device)
        all_patches = visible_epoches_data[batch_indices, epoch_indices, unmasked_patches_idx_batch]

        # Reshape for embedding: [B, num_visible_epoches * num_visible_patches, D]
        all_patches = all_patches.view(b, -1, all_patches.shape[-1])

        # 3. Patch embedding
        patch_tokens = self.patch_embedding(all_patches)

        # 4. 添加全局CLS token
        global_cls_tokens = self.global_cls_token.expand(b, -1, -1)
        global_input = torch.cat([global_cls_tokens, patch_tokens], dim=1)

        # 5. 添加位置编码
        # 计算可见patches的绝对位置
        # absolute_positions: [B, num_visible_epoches, num_visible_patches]
        absolute_positions = unmasked_epoches_index_batch.unsqueeze(2) * num_patches + unmasked_patches_idx_batch
        absolute_positions = absolute_positions.view(b, -1)

        # 创建位置编码
        max_position = num_epoches * num_patches
        full_pos_encoding = self.create_sinusoidal_encoding(max_position + 1, self.embed_dim).to(eeg.device)

        # CLS token位置编码
        cls_pos_encoding = full_pos_encoding[0:1].expand(b, -1, -1)

        # 可见patches位置编码
        patch_pos_encoding = full_pos_encoding[absolute_positions + 1]

        # 拼接位置编码
        pos_encoding = torch.cat([cls_pos_encoding, patch_pos_encoding], dim=1)
        global_input = global_input + pos_encoding

        # 6. 全局Transformer编码
        global_hidden = self.global_encoder(global_input)
        global_hidden = self.global_encoder_norm(global_hidden)

        # 7. 提取全局prompt token
        prompt_token = global_hidden[:, 0, :]

        # 生成patch_masks_dict (为了兼容后续函数)
        patch_masks_dict = {}
        for i in range(b):
            sample_masks = {}
            for j, epoch_idx in enumerate(unmasked_epoches_index_batch[i]):
                sample_masks[epoch_idx.item()] = unmasked_patches_idx_batch[i, j]
            patch_masks_dict[i] = sample_masks

        return prompt_token, unmasked_epoches_index_batch, masked_epoches_index_batch, patch_masks_dict
    
    def intra_epoch_encoding(self, eeg, prompt_token, unmasked_epoches_index_batch, patch_masks_dict):
        """
        第二阶段：基于prompt token进行intra-epoch建模 (向量化版本)
        批量处理所有可见epoch，提高计算效率

        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param prompt_token: 全局prompt token [B, embed_dim]
        :param unmasked_epoches_index_batch: 可见epochs索引 [B, num_visible_epoches]
        :param patch_masks_dict: 每个样本每个epoch的patch mask信息 {sample_idx: {epoch_idx: unmasked_patches_idx}}
        :return: epoch_cls_tokens [B, num_visible_epoches, embed_dim]
        """
        b, _, num_patches, _ = eeg.shape
        num_visible_epoches = unmasked_epoches_index_batch.shape[1]

        # 1. 批量提取所有可见epoch的可见patches
        batch_indices = torch.arange(b).view(-1, 1).to(eeg.device)
        visible_epoches_data = eeg[batch_indices, unmasked_epoches_index_batch]

        # 从 patch_masks_dict 恢复 unmasked_patches_idx_batch
        unmasked_patches_idx_batch = torch.stack(
            [torch.stack([patch_masks_dict[i][epoch_idx.item()] for epoch_idx in unmasked_epoches_index_batch[i]]) for i in range(b)]
        ).to(eeg.device)

        batch_indices = torch.arange(b).view(b, 1, 1).to(eeg.device)
        epoch_indices = torch.arange(num_visible_epoches).view(1, num_visible_epoches, 1).to(eeg.device)
        visible_patches = visible_epoches_data[batch_indices, epoch_indices, unmasked_patches_idx_batch]

        # 2. 批量进行patch embedding
        reshaped_patches = visible_patches.view(b * num_visible_epoches, -1, visible_patches.shape[-1])
        patch_tokens = self.patch_embedding(reshaped_patches)

        # 3. 构建输入序列
        epoch_cls_tokens_expanded = self.epoch_cls_token.expand(b * num_visible_epoches, -1, -1)
        prompt_tokens_expanded = prompt_token.unsqueeze(1).expand(-1, num_visible_epoches, -1).reshape(b * num_visible_epoches, 1, -1)
        intra_input = torch.cat([epoch_cls_tokens_expanded, prompt_tokens_expanded, patch_tokens], dim=1)

        # 4. 添加位置编码
        max_patches = num_patches
        full_pos_encoding = self.create_sinusoidal_encoding(max_patches + 2, self.embed_dim).to(eeg.device)

        cls_pos = full_pos_encoding[0:1].expand(b * num_visible_epoches, -1, -1)
        prompt_pos = full_pos_encoding[1:2].expand(b * num_visible_epoches, -1, -1)

        patch_pos_indices = unmasked_patches_idx_batch.view(b * num_visible_epoches, -1)
        patch_pos = full_pos_encoding[patch_pos_indices + 2]

        pos_encoding = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)
        intra_input = intra_input + pos_encoding

        # 5. 批量Transformer编码
        intra_hidden = self.intra_encoder(intra_input)
        intra_hidden = self.intra_encoder_norm(intra_hidden)

        # 6. 提取epoch CLS tokens
        epoch_cls_outputs = intra_hidden[:, 0, :]

        # 7. 重塑回原始batch维度
        epoch_cls_tokens = epoch_cls_outputs.view(b, num_visible_epoches, self.embed_dim)

        return epoch_cls_tokens
    
    def inter_epoch_encoding(self, epoch_cls_tokens, prompt_token, unmasked_epoches_index_batch, masked_epoches_index_batch):
        """
        第三阶段：inter-epoch上下文学习（使用Bi-GRU）(向量化版本)
        将prompt_token与每个epoch的CLS tokens拼接，通过Bi-GRU进行序列建模

        :param epoch_cls_tokens: epoch级特征 [B, num_visible_epoches, embed_dim]
        :param prompt_token: 全局prompt token [B, embed_dim]
        :param unmasked_epoches_index_batch: 可见epochs索引
        :param masked_epoches_index_batch: 遮蔽epochs索引
        :return: final_epoch_features [B, total_epoches, 2*embed_dim]
        """
        b, num_visible_epoches, _ = epoch_cls_tokens.shape
        num_masked_epoches = masked_epoches_index_batch.shape[1]
        total_epoches = num_visible_epoches + num_masked_epoches

        # 创建完整的epoch序列
        full_epoch_sequences = torch.zeros(b, total_epoches, self.embed_dim, device=epoch_cls_tokens.device)
        batch_indices = torch.arange(b).view(-1, 1).to(epoch_cls_tokens.device)
        full_epoch_sequences[batch_indices, unmasked_epoches_index_batch] = epoch_cls_tokens

        # 扩展prompt_token到每个epoch
        prompt_expanded = prompt_token.unsqueeze(1).expand(b, total_epoches, self.embed_dim)

        # 拼接prompt_token和epoch_cls_tokens
        inter_input = torch.cat([prompt_expanded, full_epoch_sequences], dim=2)

        # 双向GRU实现
        gru_output, _ = self.inter_bigru(inter_input)

        # 层归一化
        final_epoch_features = self.inter_norm(gru_output)

        return final_epoch_features

    def forward(self, eeg, temperature, transition=None):
        """
        新的三阶段MAE前向传播 (向量化版本)
        1. 生成全局prompt token
        2. 基于prompt token进行intra-epoch建模
        3. 进行inter-epoch上下文学习和分类

        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param temperature: 温度参数（无用懒得删）
        :param transition: transition标签 [B, num_epoches]
        :return: (sleep_stage_pred, transition_pred, transition_label)
        """
        b, num_epoches, _, _ = eeg.shape

        # 第一阶段
        prompt_token, unmasked_epoches_index_batch, masked_epoches_index_batch, patch_masks_dict = self.generate_global_prompt(eeg)

        # 第二阶段
        epoch_cls_tokens = self.intra_epoch_encoding(eeg, prompt_token, unmasked_epoches_index_batch, patch_masks_dict)

        # 第三阶段
        final_epoch_features = self.inter_epoch_encoding(epoch_cls_tokens, prompt_token,
                                                         unmasked_epoches_index_batch, masked_epoches_index_batch)

        # 分类任务
        # 1. 睡眠阶段分类
        all_epoch_features = final_epoch_features.contiguous().view(b * num_epoches, -1)
        sleep_stage_pred = self.sleep_stage_classifier(all_epoch_features)

        # 2. Transition分类 (MAE任务)
        num_masked_epoches = masked_epoches_index_batch.shape[1]
        if num_masked_epoches > 0:
            batch_indices = torch.arange(b).view(-1, 1).to(eeg.device)
            masked_features = final_epoch_features[batch_indices, masked_epoches_index_batch]
            masked_features = masked_features.view(b * num_masked_epoches, -1)
            transition_pred = self.transition_classifier(masked_features)

            if transition is not None:
                transition_label = transition[batch_indices, masked_epoches_index_batch].view(-1)
            else:
                transition_label = None
        else:
            transition_pred = torch.empty(0, 2, device=eeg.device)
            transition_label = torch.empty(0, device=eeg.device) if transition is not None else None

        return sleep_stage_pred, transition_pred, transition_label
