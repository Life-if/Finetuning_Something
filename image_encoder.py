import torch
import torch.nn as nn
from .segment_anything.modeling.sam import Sam
from .segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from .utils import fix_params
from typing import List

class BaseImgEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseImgEncodeAdapter, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x
    

class LORA(nn.Module):
    """
    实现 LoRA（Low-Rank Adaptation）模块
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        r: LoRA 的秩，控制低秩矩阵的大小
        lora_alpha: 缩放因子，通常设为 r 的倍数
        lora_dropout: Dropout 比例，用于防止过拟合
    """
    def __init__(self, in_features, out_features, r=4, lora_alpha=16, lora_dropout=0.0):
        super(LORA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else None

        # 缩放
        self.scaling = self.lora_alpha / self.r

        # 初始化 A 和 B
        nn.init.kaiming_uniform_(self.lora_A, a=1/self.r**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始线性变换由外部完成
        # LoRA 增量：x @ A @ B
        lora = x @ self.lora_A @ self.lora_B
        if self.dropout is not None:
            lora = self.dropout(lora)
        return lora * self.scaling


class LoRAAttention(nn.Module):
    """对 Attention 层应用 LoRA，仅微调 q 和 v 的投影层"""
    def __init__(self, 
                 orig_attn: nn.Module, 
                 rank: int = 4,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.orig_attn = orig_attn
        dim3 = orig_attn.qkv.out_features // 3


        # LoRA 低秩矩阵：A (dim, r), B (r, dim)
        self.lora_q = LORA(dim3,dim3,rank,alpha,dropout)
        self.lora_v = LORA(dim3,dim3,rank,alpha,dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.orig_attn.qkv(x)     # B H W C*3
        
        lora_q = self.lora_q(x)     # B H W C
        lora_v = self.lora_v(x)     # B H W C

        qkv = qkv.reshape(B, H * W, 3, self.orig_attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        # # q, k, v with shape (B, nHead, H * W, C//nHead)
        lora_q = lora_q.reshape(B, H * W, self.orig_attn.num_heads, -1).permute(0, 2, 1, 3)
        lora_v = lora_v.reshape(B, H * W, self.orig_attn.num_heads, -1).permute(0, 2, 1, 3)
        
        q, k, v = qkv.unbind(dim=0)

        # 应用 LoRA 到 q 和 v
        q = q + lora_q
        v = v + lora_v
        
        q, k, v = q.reshape(B * self.orig_attn.num_heads, H * W, -1).permute(0, 2, 1),\
            k.reshape(B * self.orig_attn.num_heads, H * W, -1).permute(0, 2, 1),\
            v.reshape(B * self.orig_attn.num_heads, H * W, -1).permute(0, 2, 1)

        attn = (q * self.orig_attn.scale) @ k.transpose(-2, -1)
        # 注释掉未定义的函数调用
        if self.orig_attn.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.orig_attn.rel_pos_h, self.orig_attn.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        x = self.orig_attn.proj(x)
        return x



class LoRAImgEncodeAdapter(BaseImgEncodeAdapter):
    """
    LoRA 适配器，用于对 SAM 的 image_encoder 进行低秩微调。
    只训练 LoRA 参数，冻结原始权重。
    """

    def __init__(self, 
                 ori_sam: Sam, 
                 rank: int = 4, 
                 alpha: float = 16,
                 lora_list: list = [1,2,4,5,7,8,10,11],
                 freeze_neck: bool = True):
        
        super(LoRAImgEncodeAdapter, self).__init__(ori_sam, fix=True)

        if not freeze_neck: 
            for name, param in self.sam_img_encoder.neck.named_parameters():
                 param.requires_grad = True
                 
        # LoRA 参数：A 和 B 矩阵，用于近似权重更新 ΔW = BA
        self.rank = rank
        self.alpha = alpha
        embed_dim = self.sam_img_encoder.patch_embed.proj.out_channels  # 如 768 (ViT-B)

        # 只对指定层进行 LoRA 微调
        self.lora_list = lora_list
        self.lora_module = nn.ModuleList()
        
        for i in self.lora_list:
            self.lora_module.append(LoRAAttention(self.sam_img_encoder.blocks[i].attn, rank=rank, lora_alpha=alpha))


    def forward(self, x):
        # 原始 patch embedding 输出
        with torch.no_grad():
            x_orig = self.sam_img_encoder.patch_embed(x)  # [B, H*W, C]

            if self.sam_img_encoder.pos_embed is not None:
                x_orig = x_orig + self.sam_img_encoder.pos_embed
        
        for idx, blk in enumerate(self.sam_img_encoder.blocks):
            if idx in self.lora_list:
                # idx在self.lora_list中的位置
                lora_idx = self.lora_list.index(idx)
                if lora_idx < len(self.lora_module):
                    # 应用对应的LoRA模块
                    x_orig = self.lora_module[lora_idx](x_orig)
                else:
                    raise ValueError(f"Invalid lora_idx: {lora_idx}")
            else:
                x_orig = blk(x_orig)
            
        x = self.sam_img_encoder.neck(x_orig)
        return x
