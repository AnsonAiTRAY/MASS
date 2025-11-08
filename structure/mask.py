import random
import torch
import torch.nn as nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio, seed=None):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.seed = seed
        # 计算需要遮蔽的token数量，使用向上取整确保可见token数量一致
        # 遮蔽数量 = ceil(num_tokens * mask_ratio)
        # 可见数量 = num_tokens - 遮蔽数量 = num_tokens - ceil(num_tokens * mask_ratio)
        import math
        self.mask_len = math.ceil(self.num_tokens * self.mask_ratio)
        # 如果提供了种子，创建一个生成器
        if self.seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

    def uniform_rand(self, batch_size=1):
        """
        生成mask索引
        :param batch_size: batch大小，为每个样本生成不同的mask
        :return: unmasked_tokens, masked_tokens - 每个都是shape为[batch_size, num_tokens]的tensor
        """
        # 统一处理，始终返回2维张量 [batch_size, num_tokens]
        unmasked_tokens_list = []
        masked_tokens_list = []
        
        for _ in range(batch_size):
            if self.seed is not None:
                indices = torch.randperm(self.num_tokens, generator=self.generator)
            else:
                indices = torch.randperm(self.num_tokens)
            masked_tokens = indices[:self.mask_len]
            unmasked_tokens = indices[self.mask_len:]
            unmasked_tokens_list.append(unmasked_tokens)
            masked_tokens_list.append(masked_tokens)
        
        # 将列表转换为tensor，shape: [batch_size, num_unmasked/masked_tokens]
        unmasked_tokens_batch = torch.stack(unmasked_tokens_list, dim=0)
        masked_tokens_batch = torch.stack(masked_tokens_list, dim=0)
        
        return unmasked_tokens_batch, masked_tokens_batch

    def forward(self, batch_size=1):
        return self.uniform_rand(batch_size)




# mask = MaskGenerator(30, 0.8)
# unmasked_token_index, masked_token_index = mask()
# print(masked_token_index, unmasked_token_index)

# unmasked_patches_index, masked_patches_index = [], []
# for x in unmasked_token_index:
#     unmasked_patches_index.extend(range(5 * x, 5 * x + 5))
# for x in masked_token_index:
#     masked_patches_index.extend(range(5 * x, 5 * x + 5))
# print(masked_patches_index, unmasked_patches_index)
