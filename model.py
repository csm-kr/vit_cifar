import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)  # product of v times whatever inside softmax
        out = self.attn_dropout(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)
        self.head_dim = int(dim // num_heads)

        # self.wQ = nn.Linear(self.dim, self.dim)
        # self.wK = nn.Linear(self.dim, self.dim)
        # self.wV = nn.Linear(self.dim, self.dim)

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
        self.multi_head_dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        xs = torch.split(x, self.head_dim, dim=-1)  # [B, 16, 384] -> tuple{(V,nor 16, 32)} * 12
        head = []
        for i in range(self.num_heads):
            q = self.wQs[i](xs[i])
            k = self.wKs[i](xs[i])
            v = self.wVs[i](xs[i])
            head.append(self.attention(q, k, v))    # [2, 16, 32]
        head = torch.cat(head, dim=-1)              # [2, 16, 384]
        x = self.wO(head)                           # [2, 16, 384]
        x = self.multi_head_dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_ratio=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.multi_head_dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads   # b: batch, n : length, h : head
        qkv = self.to_qkv(x)
        assert self.dim % self.num_heads == 0, 'dim 은 반드시 head 로 나누어 떨어져야 한다. {} % {} = {}'.\
            format(self.dim, self.num_heads, self.dim % self.num_heads)
        h_d = int(self.dim / self.num_heads)
        qkv_ = qkv.permute(2, 0, 1).view(3, h, h_d, b, n).permute(0, 3, 1, 4, 2)
        q, k, v = qkv_[0], qkv_[1], qkv_[2]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)  #
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * h_d)
        out = self.nn1(out)
        out = self.multi_head_dropout(out)
        return out


class MultiOrderedAttention(nn.Module):
    def __init__(self, dim, num_order, dropout_ratio=0.1):
        super().__init__()
        self.num_order = num_order
        self.moat = nn.ModuleList()
        for i in range(num_order):
            self.moat.append(MultiHeadAttention(dim, num_heads=4, dropout_ratio=dropout_ratio))

        # mot1 = MultiHeadAttention(dim, num_heads=4, dropout_ratio=dropout_ratio)
        # mot2 = MultiHeadAttention(dim, num_heads=4, dropout_ratio=dropout_ratio)
        # mot3 = MultiHeadAttention(dim, num_heads=4, dropout_ratio=dropout_ratio)
        # mot4 = MultiHeadAttention(dim, num_heads=4, dropout_ratio=dropout_ratio)

    def forward(self, x, num_patches=16):
        x_ = x[:, 0].unsqueeze(1)
        x = x[:, 1:, ...]
        self.num_patches = num_patches
        num_one_side_elements = int(self.num_patches ** 0.5)
        x0 = torch.cat([x_, x], dim=1)
        new_order_indices = torch.flip(torch.arange(self.num_patches).unsqueeze(0), dims=(0, 1)).squeeze()
        # print(new_order_indices) tensor([15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
        x1 = x[:, new_order_indices, :] # .contiguous()
        x1 = torch.cat([x_, x1], dim=1)
        # **  vertical order (from up to down) **
        new_order_indices = torch.arange(self.num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        # print(new_order_indices) tensor([ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15])
        x2 = x[:, new_order_indices, :]
        x2 = torch.cat([x_, x2], dim=1)
        # **  vertical order (from down to up) **
        new_order_indices = torch.arange(self.num_patches).view(num_one_side_elements, num_one_side_elements).permute(1, 0).contiguous().view(-1)
        new_order_indices = torch.flip(new_order_indices.unsqueeze(0), dims=(0, 1)).squeeze()
        # print(new_order_indices) tensor([15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0])
        x3 = x[:, new_order_indices, :]
        x3 = torch.cat([x_, x3], dim=1)
        x_list = [x0, x1, x2, x3]
        out = []
        for i in range(self.num_order):
            out.append(self.moat[i](x_list[i]))
        x = out[0] + out[3] + out[2] + out[3]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim:int, mlp_dim:int, num_heads:int=12, dropout_ratio:float=0.1):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(dim)
        # self.msa = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.msa = MultiHeadAttention(dim, num_heads=num_heads, dropout_ratio=dropout_ratio)
        # self.moa = MultiOrderedAttention(dim, num_order=4, dropout_ratio=dropout_ratio)
        self.la2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, dim),
            # nn.GELU(),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        # out = self.moa(self.la1(x)) + x
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
        torch.nn.init.normal_(self.position_embedding, std=.02)  # 확인해보기
        encoder_list = [TransformerEncoder(dim,
                                           mlp_dim=mlp_dim,
                                           dropout_ratio=dropout_ratio,
                                           num_heads=num_heads) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*encoder_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.pos_dropout = nn.Dropout(dropout_ratio)
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
        x = self.pos_dropout(x)
        x = self.transformer(x)                                                      # [B, 257, D]
        x = x[:, 0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vit = ViT(patch_size=4,
              image_size=32,
              num_layers=7,
              dim=384,
              mlp_dim=384,
              num_heads=4,
              dropout_ratio=0.1,
              num_classes=10,
              is_cls_token=True)

    # transformer = TransformerEncoder(dim=384, mlp_dim=384, num_heads=12, dropout_ratio=0.1)
    # msa = MultiHeadSelfAttention(dim=384, num_heads=12)
    x = torch.randn([2, 3, 32, 32])
    x = vit(x)
    print(x.size())