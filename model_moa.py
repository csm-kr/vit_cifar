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

        # ===================== new positional embedding =====================
        # self.position_embedding_ = nn.Parameter(torch.empty([1, 1, image_size // patch_size, image_size // patch_size]))
        # torch.nn.init.normal_(self.position_embedding_, std=.2)
        # self.pos_embed_conv = nn.Conv2d(1, self.dim, kernel_size=2, stride=2, padding=0)
        # self.position_embedding = nn.Parameter(self.make_conv_position_embedding())
        # ===================== new positional embedding =====================

    # def make_conv_position_embedding(self):
    #     pos_embedding_conv = self.pos_embed_conv(self.position_embedding_)  # [1, 384, 8, 8]
    #     pos_embedding_conv = torch.nn.functional.interpolate(pos_embedding_conv, (8, 8))
    #     pos_embedding_conv = pos_embedding_conv.permute(0, 2, 3, 1).view(1, self.num_patches, self.dim)
    #     return pos_embedding_conv

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding_projection(x)                                                # [B, 64, 28, 28]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_patches, self.dim)           # [B, L, D]
        if self.is_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)                            # Expand to [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1)                                             # [B, 1 + N, D]
            # cls_token [cls_token; x_p^1E; x_p^2E; ..., ;x_p^N]

        # else [x_p^1E; x_p^2E; ..., ;x_p^N]

        # pos_embedding_conv = self.pos_embed_conv(self.position_embedding_)  # [1, 384, 8, 8]
        # pos_embedding_conv = torch.nn.functional.interpolate(pos_embedding_conv, (8, 8)).permute(0, 2, 3, 1)
        # ============================= visualize positional embedding =============================
        # vis_pos_embedding = self.position_embedding.squeeze(0)\
        #     .view(int(self.num_patches ** 0.5), int(self.num_patches ** 0.5), self.dim)  # 8, 8, 384
        # vis_pos_embedding = vis_pos_embedding.mean(dim=-1)
        # vis_pos_embedding /= vis_pos_embedding.max()
        # vis_pos_embedding = vis_pos_embedding.detach().cpu()
        #
        # import matplotlib.pyplot as plt
        # plt.figure('mean pos_embedding')
        # plt.imshow(vis_pos_embedding)
        # plt.show()
        # ============================= visualize positional embedding =============================

        x += self.position_embedding
        x = self.pos_dropout(x)
        return x


class MultiOrderedAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio, num_orders=4):
        super().__init__()

        self.num_orders = num_orders
        self.order1_nn = nn.Linear(dim, dim // num_orders, bias=True)
        self.order2_nn = nn.Linear(dim, dim // num_orders, bias=True)
        self.order3_nn = nn.Linear(dim, dim // num_orders, bias=True)
        self.order4_nn = nn.Linear(dim, dim // num_orders, bias=True)

        self.order_att1 = MultiHeadAttention(dim=dim//num_orders, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.order_att2 = MultiHeadAttention(dim=dim//num_orders, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.order_att3 = MultiHeadAttention(dim=dim//num_orders, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.order_att4 = MultiHeadAttention(dim=dim//num_orders, num_heads=num_heads, dropout_ratio=dropout_ratio)

        self.out_nn = nn.Linear(dim, dim, bias=True)

    def perm_orders(self, x):
        num_patches = x.size(1)
        x1 = x
        num_one_side_elements = int(num_patches ** 0.5)
        # ** reverse order **
        new_order_indices = torch.flip(torch.arange(num_patches).unsqueeze(0), dims=(0, 1)).squeeze()
        # print(new_order_indices) tensor([15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
        x2 = x[:, new_order_indices, :]
        # **  vertical order (from up to down) **
        new_order_indices = torch.arange(num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        # print(new_order_indices) tensor([ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15])
        x3 = x[:, new_order_indices, :]
        # **  vertical order (from down to up) **
        new_order_indices = torch.arange(num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        new_order_indices = torch.flip(new_order_indices.unsqueeze(0), dims=(0, 1)).squeeze()
        # print(new_order_indices) tensor([15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0])
        x4 = x[:, new_order_indices, :]
        return [x1, x2, x3, x4]

    def forward(self, x):
        x1, x2, x3, x4 = self.perm_orders(x)

        x1 = self.order1_nn(x1)
        x2 = self.order1_nn(x2)
        x3 = self.order1_nn(x3)
        x4 = self.order1_nn(x4)

        x1, _ = self.order_att1(x1)
        x2, _ = self.order_att2(x2)
        x3, _ = self.order_att3(x3)
        x4, _ = self.order_att4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.out_nn(x)
        return x, 0


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
        self.msa = MultiOrderedAttention(dim, num_heads, dropout_ratio)
        # self.msa = MultiHeadAttention(dim, num_heads, dropout_ratio)
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
              patch_size=4, image_size=32, is_cls_token=False,
              dropout_ratio=0.1, num_classes=10)

    x = vit(x, False)
    # print(x.size())
    # print(attn_mask.size())
    #
    # attn_mask[0] /= attn_mask[0].max()
    # attn_mask_numpy_batch_0 = attn_mask[0].detach().cpu().numpy()
    # import cv2
    # attn_mask_numpy_batch_0 = cv2.resize(attn_mask_numpy_batch_0, (100, 100))
    # cv2.imshow('input', attn_mask_numpy_batch_0)
    # cv2.waitKey()

    x = torch.randn([3, 64, 384])
    moa = MultiOrderedAttention(dim=384, num_heads=12, dropout_ratio=0.0)
    print(moa(x).size())

