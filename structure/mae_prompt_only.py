import torch
from torch import nn
import torch.nn.functional as F
from .transformer_layers import TransformerLayers
import math


class SleepMAEPromptOnly(nn.Module):
    """消融实验 - Prompt Only版本：仅使用Prompt机制，无Mask
    
    该版本保留了prompt机制但移除了masking机制，用于验证Prompt是否能引导建模。
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, num_patches, num_epoches, intra_mask_ratio,
                 inter_mask_ratio, global_encoder_depth, intra_encoder_depth, inter_encoder_depth, inter_hidden_dim, 
                 mean, std, use_cls_token=True, seed=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.global_encoder_depth = global_encoder_depth
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
        
        # 全局CLS token - 用于生成全局prompt token
        # 维度: [1, embed_dim]，在所有patches前添加
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # epoch级CLS token - 用于每个epoch内的特征聚合
        # 维度: [1, embed_dim]，在每个epoch的patches前添加
        self.epoch_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 第一阶段：全局Transformer encoder - 用于生成prompt token
        # 输入：所有patches + global_cls_token（无masking）
        # 输出：global prompt token
        self.global_encoder = TransformerLayers(embed_dim, global_encoder_depth, mlp_ratio, num_heads, dropout)
        self.global_encoder_norm = nn.LayerNorm(embed_dim)
        
        # 第二阶段：intra-epoch Transformer encoder
        # 输入：每个epoch的所有patches + prompt_token + epoch_cls_token（无masking）
        # 输出：每个epoch的CLS token作为epoch级特征
        self.intra_encoder = TransformerLayers(embed_dim, intra_encoder_depth, mlp_ratio, num_heads, dropout)
        self.intra_encoder_norm = nn.LayerNorm(embed_dim)
        
        # 第三阶段：inter-epoch Bi-GRU encoder（与完整MAE模型保持一致）
        # 输入：拼接后的特征 [B, total_epoches, 2*embed_dim] (prompt_token + epoch_cls_token)
        # 输出：每个epoch的最终特征表示 [B, total_epoches, 2*inter_hidden_dim]
        self.inter_bigru = nn.GRU(
            input_size=2*embed_dim,  # 输入维度为2*embed_dim（prompt + epoch_cls拼接）
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
        
        # 测试时mask：使用固定比例进行mask
        test_intra_ratio = self.intra_mask_ratio  # 测试时patch mask比例
        test_inter_ratio = self.inter_mask_ratio  # 测试时epoch mask比例
        
        # 1. Inter-epoch masking: 随机mask一些epochs
        if test_inter_ratio > 0:
            num_masked_epochs = int(num_epoches * test_inter_ratio)
            for batch_idx in range(b):
                # 为每个样本随机选择要mask的epochs
                masked_epoch_indices = torch.randperm(num_epoches)[:num_masked_epochs]
                # 将选中的epochs设置为0
                masked_eeg[batch_idx, masked_epoch_indices, :, :] = 0
        
        # 2. Intra-epoch masking: 在未被epoch-level mask的epochs中随机mask一些patches
        if test_intra_ratio > 0:
            num_masked_patches = int(num_patches * test_intra_ratio)
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

    def generate_global_prompt(self, eeg):
        """
        第一阶段：生成全局prompt token
        从所有patches中学习全局上下文信息（无masking）
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :return: prompt_token [B, embed_dim] - 全局prompt token
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 1. 使用所有patches（无masking机制）
        # 重塑为: [B, num_epoches * num_patches, 128]
        all_patches = eeg.view(b, num_epoches * num_patches, 128)
        
        # 2. Patch embedding: [B, num_epoches * num_patches, embed_dim]
        patch_tokens = self.patch_embedding(all_patches)
        
        # 3. 添加全局CLS token: [B, 1 + num_epoches * num_patches, embed_dim]
        global_cls_tokens = self.global_cls_token.expand(b, -1, -1)  # [B, 1, embed_dim]
        global_input = torch.cat([global_cls_tokens, patch_tokens], dim=1)
        
        # 4. 添加位置编码（根据原始序列位置）
        max_position = num_epoches * num_patches  # 最大可能的位置索引
        full_pos_encoding = self.create_sinusoidal_encoding(max_position + 1, self.embed_dim).to(eeg.device)  # +1 for CLS
        
        # 为每个样本构建位置编码
        batch_pos_encoding = []
        for i in range(b):
            # CLS token使用位置0
            sample_pos_encoding = [full_pos_encoding[0:1]]  # [1, embed_dim]
            # 所有patches使用其在原始序列中的绝对位置
            for pos in range(num_epoches * num_patches):
                sample_pos_encoding.append(full_pos_encoding[pos + 1:pos + 2])  # +1 因为CLS占用位置0
            # 拼接: [1 + num_epoches * num_patches, embed_dim]
            sample_pos_encoding = torch.cat(sample_pos_encoding, dim=0)
            batch_pos_encoding.append(sample_pos_encoding)
        
        # 堆叠所有样本: [B, 1 + num_epoches * num_patches, embed_dim]
        pos_encoding = torch.stack(batch_pos_encoding, dim=0)
        global_input = global_input + pos_encoding
        
        # 5. 全局Transformer编码
        # 输入: [B, 1 + num_epoches * num_patches, embed_dim]
        # 输出: [B, 1 + num_epoches * num_patches, embed_dim]
        global_hidden = self.global_encoder(global_input)
        global_hidden = self.global_encoder_norm(global_hidden)
        
        # 6. 提取全局prompt token (CLS token的输出)
        prompt_token = global_hidden[:, 0, :]  # [B, embed_dim]
        
        return prompt_token
    
    def intra_epoch_encoding(self, eeg, prompt_token):
        """
        第二阶段：基于prompt token进行intra-epoch建模
        使用所有patches（无masking）
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param prompt_token: 全局prompt token [B, embed_dim]
        :return: epoch_cls_tokens [B, num_epoches, embed_dim]
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 1. 批量进行patch embedding
        # 重塑为: [B * num_epoches, num_patches, 128]
        reshaped_patches = eeg.view(b * num_epoches, num_patches, 128)
        
        # Patch embedding: [B * num_epoches, num_patches, embed_dim]
        patch_tokens = self.patch_embedding(reshaped_patches)
        
        # 2. 构建输入序列：epoch_cls_token + prompt_token + patch_tokens
        # 扩展epoch_cls_token: [1, 1, embed_dim] -> [B * num_epoches, 1, embed_dim]
        epoch_cls_tokens_expanded = self.epoch_cls_token.expand(b * num_epoches, -1, -1)
        
        # 扩展prompt_token: [B, embed_dim] -> [B * num_epoches, 1, embed_dim]
        prompt_tokens_expanded = prompt_token.unsqueeze(1).repeat(1, num_epoches, 1)  # [B, num_epoches, embed_dim]
        prompt_tokens_expanded = prompt_tokens_expanded.view(b * num_epoches, 1, self.embed_dim)  # [B * num_epoches, 1, embed_dim]
        
        # 拼接所有token: [B * num_epoches, 2 + num_patches, embed_dim]
        intra_input = torch.cat([epoch_cls_tokens_expanded, prompt_tokens_expanded, patch_tokens], dim=1)
        
        # 3. 添加位置编码（根据原始epoch内序列位置）
        full_pos_encoding = self.create_sinusoidal_encoding(num_patches + 2, self.embed_dim).to(eeg.device)  # +2 for CLS and prompt
        
        # 为每个epoch构建位置编码
        batch_pos_encoding = []
        for i in range(b * num_epoches):
            # 构建该epoch的位置编码序列
            epoch_pos_encoding = []
            # CLS token使用位置0
            epoch_pos_encoding.append(full_pos_encoding[0:1])  # [1, embed_dim]
            # Prompt token使用位置1
            epoch_pos_encoding.append(full_pos_encoding[1:2])  # [1, embed_dim]
            # 所有patches使用其在原始epoch内的位置（+2因为CLS和prompt占用前两个位置）
            for patch_pos in range(num_patches):
                epoch_pos_encoding.append(full_pos_encoding[patch_pos + 2:patch_pos + 3])
            
            # 拼接该epoch的位置编码: [2 + num_patches, embed_dim]
            epoch_pos_encoding = torch.cat(epoch_pos_encoding, dim=0)
            batch_pos_encoding.append(epoch_pos_encoding)
        
        # 堆叠所有epoch的位置编码: [B * num_epoches, 2 + num_patches, embed_dim]
        pos_encoding = torch.stack(batch_pos_encoding, dim=0)
        intra_input = intra_input + pos_encoding
        
        # 4. 批量Transformer编码（并行处理所有epoch）
        # 输入: [B * num_epoches, 2 + num_patches, embed_dim]
        # 输出: [B * num_epoches, 2 + num_patches, embed_dim]
        intra_hidden = self.intra_encoder(intra_input)
        intra_hidden = self.intra_encoder_norm(intra_hidden)
        
        # 5. 提取epoch CLS tokens
        # [B * num_epoches, embed_dim]
        epoch_cls_outputs = intra_hidden[:, 0, :]
        
        # 6. 重塑回原始batch维度: [B, num_epoches, embed_dim]
        epoch_cls_tokens = epoch_cls_outputs.view(b, num_epoches, self.embed_dim)
        
        return epoch_cls_tokens
    
    def inter_epoch_encoding(self, epoch_cls_tokens, prompt_token):
        """
        第三阶段：inter-epoch上下文学习
        将prompt_token与每个epoch的CLS tokens拼接，通过Bi-GRU进行序列建模（与完整MAE模型保持一致）
        
        :param epoch_cls_tokens: epoch级特征 [B, num_epoches, embed_dim]
        :param prompt_token: 全局prompt token [B, embed_dim]
        :return: final_epoch_features [B, num_epoches, 2*inter_hidden_dim]
        """
        b, num_epoches, _ = epoch_cls_tokens.shape
        
        # 扩展prompt_token到每个epoch: [B, num_epoches, embed_dim]
        # 每个epoch都与相同的全局prompt_token拼接
        prompt_expanded = prompt_token.unsqueeze(1).expand(b, num_epoches, self.embed_dim)
        
        # 拼接prompt_token和epoch_cls_tokens: [B, num_epoches, 2*embed_dim]
        # 维度变为原来的两倍，包含全局上下文信息和epoch特定信息
        inter_input = torch.cat([prompt_expanded, epoch_cls_tokens], dim=2)
        
        # 通过Bi-GRU进行序列建模: [B, num_epoches, 2*embed_dim] -> [B, num_epoches, 2*inter_hidden_dim]
        # Bi-GRU能够捕获双向的时序依赖关系
        bigru_output, _ = self.inter_bigru(inter_input)
        
        return bigru_output

    def forward(self, eeg, temperature, transition=None):
        """
        Prompt Only版本前向传播：使用Prompt机制但无Mask
        
        :param eeg: 输入EEG数据 [B, num_epoches, num_patches, 128]
        :param temperature: 温度参数（保留兼容性）
        :param transition: transition标签 [B, num_epoches]
        :return: (sleep_stage_pred, transition_pred, transition_label)
        """
        b, num_epoches, num_patches, _ = eeg.shape
        
        # 测试时应用mask（如果启用）
        if not self.training and self.test_mask_enabled:
            eeg = self.apply_test_mask(eeg)
        
        # 第一阶段：生成全局prompt token
        # 从所有patches中学习全局上下文信息（无masking）
        # prompt_token: [B, embed_dim]
        prompt_token = self.generate_global_prompt(eeg)
        
        # 第二阶段：基于prompt token进行intra-epoch建模
        # 为每个epoch添加prompt token指导学习，使用所有patches（无masking）
        # epoch_cls_tokens: [B, num_epoches, embed_dim]
        epoch_cls_tokens = self.intra_epoch_encoding(eeg, prompt_token)
        
        # 第三阶段：inter-epoch上下文学习
        # 将所有epoch的CLS tokens与prompt token一起进行序列建模
        # final_epoch_features: [B, num_epoches, 2*embed_dim]
        final_epoch_features = self.inter_epoch_encoding(epoch_cls_tokens, prompt_token)
        
        # 分类任务
        # 1. 睡眠阶段分类：对所有epochs进行分类
        # final_epoch_features: [B, num_epoches, 2*inter_hidden_dim] -> [B * num_epoches, 2*inter_hidden_dim]
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