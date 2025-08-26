import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .segment_anything.modeling.common import LayerNorm2d


class OriHead(nn.Module):
    """
    基于Transformer架构，根据图像和提示嵌入来预测掩码

    参数:
      transformer_dim (int): Transformer的通道维度
      num_multimask_outputs (int): 在消除掩码歧义时需要预测的掩码数量
      activation (nn.Module): 上采样掩码时使用的激活函数类型
      iou_head_depth (int): 用于预测掩码质量的MLP的深度
      iou_head_hidden_dim (int): 用于预测掩码质量的MLP的隐藏层维度
    """

    def __init__(
            self,
            *,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        
        super().__init__()

        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        # 计算掩码token的数量，等于多掩码输出数+1
        self.num_mask_tokens = num_multimask_outputs + 1


        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),

            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # 创建IoU预测头，用于预测掩码质量
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            src: torch.Tensor,
            iou_token_out: torch.Tensor,
            mask_tokens_out: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据图像和提示嵌入预测掩码。

        参数：
            image_embeddings (torch.Tensor)：来自图像编码器的嵌入；
            image_pe (torch.Tensor)：与图像嵌入形状相同的位置编码；
            sparse_prompt_embeddings (torch.Tensor)：点和框的嵌入；
            dense_prompt_embeddings (torch.Tensor)：掩码输入的嵌入；
            multimask_output (bool)：是否返回多个掩码或单个掩码。

        返回：
        torch.Tensor：批量预测的掩码；
        torch.Tensor：批量预测的掩码质量。
        """
 
        b, c, h, w = src.shape

        # 调整张量维度并恢复为4D张量格式
        src = src.transpose(1, 2).view(b, c, h, w)
 
        upscaled_embedding = self.output_upscaling(src)
        # 存储超网络输出的列表
        hyper_in_list: List[torch.Tensor] = []
        # 对每个掩码token应用对应的超网络MLP
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # 将超网络输出堆叠为张量
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # 获取上采样嵌入的尺寸信息
        b, c, h, w = upscaled_embedding.shape
        # 使用超网络输出和上采样嵌入计算最终的掩码
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        # 根据multimask_output参数选择输出的掩码
        if multimask_output:
            # 如果需要多个掩码，选择除第一个外的所有掩码
            mask_slice = slice(1, None)
        else:
            # 如果只需要一个掩码，选择第一个掩码
            mask_slice = slice(0, 1)
            
        # 应用切片选择对应的掩码和IoU预测
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


# SemSegHead类：用于语义分割的头部模块
class SemSegHead(nn.Module):

    def __init__(
            self,
            *,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            class_num: int = 20,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        # 调用父类初始化方法
        super().__init__()
        # 保存transformer的通道维度
        self.transformer_dim = transformer_dim
        # 保存多掩码输出数量
        self.num_multimask_outputs = num_multimask_outputs
        # 计算掩码token数量
        self.num_mask_tokens = num_multimask_outputs + 1
        # 保存类别数量
        self.class_num = class_num

        # 定义上采样模块
        self.output_upscaling = nn.Sequential(
            # 转置卷积层，用于上采样
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            # 2D层归一化
            LayerNorm2d(transformer_dim // 4),
            # 激活函数
            activation(),
            # 再一次转置卷积层进行上采样
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            # 激活函数
            activation(),
        )

        # 创建超网络MLP列表，数量等于类别数
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.class_num)
            ]
        )

        # 创建IoU预测头
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            src: torch.Tensor,
            iou_token_out: torch.Tensor,
            mask_tokens_out: torch.Tensor,
            src_shape,
            mask_scale=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          src (torch.Tensor): The tensor contains image embedding and sparse prompt embedding
          iou_token_out (torch.Tensor): Tokens of iou prediction from neck module
          mask_tokens_out (torch.Tensor): Tokens of mask prediction form neck module
          mask_scale (int): Original SAM output 3 masks which is from local to global as default
                            This Class use one of three mask tokens to transform it into class-ware
                            semantic segmentation prediction

        Returns:
          torch.Tensor: batched predicted semantic masks
          torch.Tensor: batched predictions of mask quality
        """
        # 获取输入特征形状
        b, c, h, w = src_shape

        # 调整张量维度并恢复为4D张量格式
        src = src.transpose(1, 2).view(b, c, h, w)
        # 对掩码嵌入进行上采样
        upscaled_embedding = self.output_upscaling(src)
        # 存储超网络输出的列表
        hyper_in_list: List[torch.Tensor] = []
        # 对每个类别应用对应的超网络MLP，使用指定的mask_scale索引的掩码token
        for i in range(self.class_num):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, mask_scale, :]))
        # 将超网络输出堆叠为张量
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 获取上采样嵌入的尺寸信息
        b, c, h, w = upscaled_embedding.shape
        # 使用超网络输出和上采样嵌入计算语义分割掩码，输出形状为B(批次) N(类别数) H W
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # B N H W, N is num of category

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)  # B N H W, N is num of category

        # 返回预测的语义分割掩码和IoU预测结果
        return masks, iou_pred



class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:

        super().__init__()

        self.num_layers = num_layers
        
        # 创建隐藏层维度列表，长度为num_layers-1，每个元素都是hidden_dim,
        h = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
        # 是否使用sigmoid激活函数作为输出
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # 遍历每一层网络
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)

        return x