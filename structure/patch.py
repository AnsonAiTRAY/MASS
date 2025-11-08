from torch import nn
import torch


class PatchEmbedding(nn.Module):
    """Patchify raw eeg with additional linear mapping."""

    def __init__(self, patch_size, in_channel, embed_dim):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size  # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(in_channel, embed_dim, kernel_size=(self.len_patch, 1),
                                         stride=(self.len_patch, 1))
        self.linear_mapping1 = nn.Linear(embed_dim, embed_dim)  # Linear mapping layer1
        self.linear_mapping2 = nn.Linear(embed_dim, embed_dim)  # Linear mapping layer2
        self.norm_layer = nn.GroupNorm(num_groups=4, num_channels=embed_dim)
        self.activate = nn.GELU()

    def forward(self, eeg):
        """
        Args: eeg (torch.Tensor): EEG signal with shape [B, N, 1, P * L]. P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified EEG with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = eeg.shape
        eeg = eeg.unsqueeze(-1)
        eeg = eeg.reshape(batch_size * num_nodes, num_feat, len_time_series, 1)  # [B * N, 1, P * L, 1]
        # Initial embedding
        output = self.input_embedding(eeg)
        # Activation and norm after embedding
        output = self.activate(output)
        output = self.norm_layer(output)

        # Linear mapping 1 on embed_dim
        output = output.permute(0, 2, 3, 1)  # [batch_size * num_nodes, P, 1, embed_dim]
        output = self.linear_mapping1(output)
        output = self.activate(output)
        output = output.permute(0, 3, 1, 2)
        output = self.norm_layer(output)

        # # Linear mapping 2 on embed_dim
        # output = output.permute(0, 2, 3, 1)  # [batch_size * num_nodes, P, 1, embed_dim]
        # output = self.linear_mapping2(output)
        # output = self.activate(output)
        # output = output.permute(0, 3, 1, 2)
        # output = self.norm_layer(output)

        # Reshape back and permute to original shape
        output = output.permute(0, 3, 1, 2)  # [batch_size * num_nodes, embed_dim, P, 1]
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P

        assert output.shape[-1] == len_time_series / self.len_patch
        return output


# embed = PatchEmbedding(20, 1, 128)
# eeg = torch.randn(240, 1, 1, 3000)
# out = embed(eeg)
# print(out.shape)
