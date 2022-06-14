import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, patch_size, image_size, dropout_ratio=0.1, is_cls_token=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size                                                          # 16 x 16 in ViT
        self.num_patches = (image_size // patch_size) ** 2                                    # L
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if is_cls_token else None       # [1, 1, D]
        self.patch_embedding_projection = nn.Conv2d(3, dim, patch_size, stride=patch_size)    # Non overlap projection
        self.position_embedding = nn.Parameter(torch.empty(1, (self.num_patches + 1), dim)) if is_cls_token \
            else nn.Parameter(torch.empty(1, self.num_patches, dim))                          # [1, N (+1), D]
        torch.nn.init.normal_(self.position_embedding, std=.02)

        self.pos_dropout = nn.Dropout(dropout_ratio)
        self.is_cls_token = is_cls_token

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding_projection(x)                                                # [B, 64, 28, 28]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_patches, self.dim)           # [B, L, D]
        if self.is_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)                            # Expand to [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1)                                             # [B, 1 + N, D]
            # cls_token [cls_token; x_p^1E; x_p^2E; ..., ;x_p^N]
        # else [x_p^1E; x_p^2E; ..., ;x_p^N]
        x += self.position_embedding
        x = self.pos_dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5                          # 1/sqrt(dim)

        self.q = nn.Linear(dim, dim, bias=True)           # Wq
        self.k = nn.Linear(dim, dim, bias=True)           # Wk
        self.v = nn.Linear(dim, dim, bias=True)           # Wv
        self.o = nn.Linear(dim, dim)                      # Wo

        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.multi_head_dropout = nn.Dropout(dropout_ratio)

        self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.zeros_(self.q.bias)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.zeros_(self.k.bias)
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.zeros_(self.v.bias)
        torch.nn.init.xavier_uniform_(self.o.weight)
        torch.nn.init.zeros_(self.o.bias)

    def forward(self, x):
        assert self.dim % self.num_heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)

        b, l, _, h = *x.shape, self.num_heads                          # b: batch, l : length, h : head
        h_d = int(self.dim / self.num_heads)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D
        k = k.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D
        v = v.permute(2, 0, 1).view(h, h_d, b, l).permute(2, 0, 3, 1)  # B, H, L, H_D

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale      # B, H, L, L
        attn = dots.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attn, v)                   # B, H, L, H_D
        x = self.attn_dropout(x)

        x = x.permute(0, 2, 1, 3).reshape(b, l, h * h_d)               # B, L, D
        x = self.o(x)
        x = self.multi_head_dropout(x)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.msa = MultiHeadAttention(dim, num_heads, dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_ratio),
        )
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x_, attn = self.msa(self.layer_norm1(x))
        x = x_ + x                                   # for residual
        x = self.mlp(self.layer_norm2(x)) + x
        return x, attn


class ViT(nn.Module):
    def __init__(self,
                 dim, mlp_dim, num_heads, num_layers,
                 patch_size, image_size, is_cls_token,
                 dropout_ratio, num_classes):
        super().__init__()
        self.is_cls_token = is_cls_token
        self.embedding = EmbeddingLayer(dim, patch_size, image_size, dropout_ratio, is_cls_token)
        self.encoder_list = [EncoderLayer(dim, mlp_dim, num_heads, dropout_ratio) for _ in range(num_layers)]
        self.transformer_module_list = nn.ModuleList(self.encoder_list)
        # self.transformer = nn.Sequential(*self.encoder_list)
        # self.encoder_norm = nn.LayerNorm(dim)

        # init fc
        self.fc = nn.Linear(dim, num_classes)
        self.fc.apply(self.init_weights)

        print("num_params : ", self.count_parameters())

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.bias, std=1e-6)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, is_attn=False):
        batch_size = x.size(0)
        attns = []
        x = self.embedding(x)                                       # [B, L, D]
        for module in self.transformer_module_list:
            x, attn = module(x)
            attns.append(attn)

        # x = self.transformer(x)                                     # [B, L, D]
        if self.is_cls_token:
            x = x[:, 0]                                             # [B, D]
        else:
            x = torch.mean(x, dim=1)                                # [B, D]
        # no encoder norm is better
        # x = self.encoder_norm(x)                                  # [B, D]
        x = self.fc(x)                                              # [B, num_classes]

        if is_attn:
            attn_mask = torch.stack(attns).permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, 17, 17).mean(1)  # [2, 16, 16]
            return x, attn_mask
        return x


if __name__ == '__main__':
    x = torch.randn([2, 3, 32, 32])
    vit = ViT(dim=384, mlp_dim=384, num_heads=12, num_layers=7,
              patch_size=8, image_size=32, is_cls_token=True,
              dropout_ratio=0.1, num_classes=10)

    # x, attn_mask = vit(x, True)
    # print(x.size())
    # print(attn_mask.size())
    #
    # attn_mask[0] /= attn_mask[0].max()
    # attn_mask_numpy_batch_0 = attn_mask[0].detach().cpu().numpy()
    # import cv2
    # attn_mask_numpy_batch_0 = cv2.resize(attn_mask_numpy_batch_0, (100, 100))
    # cv2.imshow('input', attn_mask_numpy_batch_0)
    # cv2.waitKey()

