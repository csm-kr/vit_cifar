import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)  # product of v times whatever inside softmax
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)
        self.head_dim = int(dim // num_heads)
        self.wQs = nn.ModuleList()
        self.wKs = nn.ModuleList()
        self.wVs = nn.ModuleList()
        self.wO = nn.Linear(self.dim, self.dim)
        self.attention = Attention(self.dim)
        for i in range(self.num_heads):
            self.wQs.append(nn.Linear(self.head_dim, self.head_dim))
            self.wKs.append(nn.Linear(self.head_dim, self.head_dim))
            self.wVs.append(nn.Linear(self.head_dim, self.head_dim))
        torch.nn.init.xavier_uniform_(self.wO.weight)
        torch.nn.init.zeros_(self.wO.bias)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        xs = torch.split(x, self.head_dim, dim=-1)  # [B, 16, 384] -> tuple{(V, 16, 32)} * 12
        head = []
        for i in range(self.num_heads):
            q = self.wQs[i](xs[i])
            k = self.wKs[i](xs[i])
            v = self.wVs[i](xs[i])
            head.append(self.attention(q, k, v))    # [2, 16, 32]
        head = torch.cat(head, dim=-1)              # [2, 16, 384]
        x = self.wO(head)                           # [2, 16, 384]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim:int, mlp_dim:int, num_heads:int=12, dropout_ratio:float=0.1):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(dim)
        self.msa = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.la2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class ViT(nn.Module):
    def __init__(self, patch_size, image_size, num_layers, dim, mlp_dim, num_heads, dropout_ratio, num_classes,
                 is_cls_token):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size                                                              # 1 // 4
        self.num_patches = (image_size // patch_size) ** 2                                        # 784
        # number of patches (N)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim)) if is_cls_token else None      # [1, 1, D]
        self.patch_embedding_projection = nn.Conv2d(3, self.dim, self.patch_size, stride=self.patch_size)  # projection
        self.position_embedding = nn.Parameter(torch.empty(1, (self.num_patches + 1), self.dim))  # [1, N + 1, D]
        encoder_list = [TransformerEncoder(dim, mlp_dim=mlp_dim, dropout_ratio=dropout_ratio, num_heads=num_heads) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*encoder_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding_projection(x)                                       # [B, 64, 28, 28]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_patches, self.dim)  # [B, 784, 64]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)                       # expand to [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)                                        # [B, 1 + N, D]
        x += self.position_embedding
        x = self.transformer(x)                                                      # [B, 257, D]
        x = x[:, 0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vit = ViT(patch_size=8, image_size=32, num_layers=7,
              dim=384, mlp_dim=384, num_heads=12,
              dropout_ratio=0.1,
              num_classes=10, is_cls_token=True)
    # transformer = TransformerEncoder(dim=384, mlp_dim=384, num_heads=12, dropout_ratio=0.1)
    # msa = MultiHeadSelfAttention(dim=384, num_heads=12)
    x = torch.randn([2, 3, 32, 32])
    x = vit(x)
    print(x.size())