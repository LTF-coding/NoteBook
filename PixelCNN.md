1. ***Pixel Recurrent Neural Networks***, DeepMind
2. ***Conditional Image Generation with PixelCNN Decoder***, DeepMind

实际上论文1共提出了两种架构：PixelRNN和PixelCNN
+ PixelRNN 使用二维LSTM (Long Short-Term Memory) 层来建模像素的条件分布。LSTM 网络擅长处理序列数据中的长距离依赖关系。
+   PixelCNN 使用CNN来建模条件分布。为了遵守自回归特性，PixelCNN 对卷积核使用了**掩码机制**，以确保模型在预测当前像素时只依赖于已经生成的上方和左侧的像素信息。
相比于RNN，CNN计算更**高效**，这是因为卷积操作本质上比 LSTM 更容易**并行化**，我们下面只讨论PixelCNN。

### 概括
+ PixelCNN 是一种基于CNN的**图像密度模型**，采用自回归的方式对图像进行逐像素的建模。
+ PixelCNN借用了NLP里的方法来生成图像。对于自然图像，每个像素值的取值范围为0~255，共256个离散值。PixelCNN模型会**根据前i - 1个像素输出第i个像素的概率分布**(多项分布)，并从预测的概率分布里**采样**出第i个像素。
+ 训练时，和多分类任务一样，要根据第i个像素的真值和预测的概率分布求交叉熵损失函数

### 关键概念
#### 像素值的建模
与之前一些使用连续分布建模像素值的尝试不同，PixelCNN/PixelRNN 将像素值**建模为离散变量**。每个通道的变量可以取 256 个不同的值。模型的条件分布通过一个 Softmax 层实现，输出一个**多项式分布**。这种离散分布方法具有表示简单、易于学习的优点，并且可以是**任意多峰的**。实验表明，这种方法比连续分布（如混合高斯模型）能获得更好的性能。

#### 掩码机制（Masking Mechanism）
下方或严格右侧的像素，这些像素在生成当前像素时是“未来”的信息，不应该被模型看到。为了解决这个问题，PixelCNN 对卷积核应用了掩码。这些掩码通过将卷积核中对应于未来**像素位置的权重设置为零**来实现这样，模型的感受野就被限制在当前像素的上方和左侧区域图 1 (中) 展示了一个 5x5 滤波器如何被掩码以确保模型无法读取当前像素下方或右侧的像素来进行预测。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
import time
import einops
import cv2
import numpy as np
import os


class MaskConv2d(nn.Module):
    """
        掩码卷积的实现思路：
            在卷积核组上设置一个mask，在前向传播的时候，先让卷积核组乘mask，再做普通的卷积
    """
    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ('A', 'B')
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        # 由于输入输出都是单通道图像，我们只需要在卷积核的h, w两个维度设置掩码
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2] = 1
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        # 为了保证掩码能正确广播到4维的卷积核组上，我们做一个reshape操作
        mask = mask.reshape((1, 1, H, W))
        # register_buffer可以把一个变量加入成员变量的同时，记录到PyTorch的Module中
        # 每当执行model.to(device)把模型中所有参数转到某个设备上时，被注册的变量会跟着转。
        # 第三个参数表示被注册的变量是否要加入state_dict中以保存下来
        self.register_buffer(name='mask', tensor=mask, persistent=False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res
```

#### 多通道
除了空间位置的依赖关系，掩码还应用于像素内部的颜色通道（R、G、B）之间在 PixelCNN 中，三个颜色通道是依次建模的。例如，蓝色通道 (B) 的建模以红色 (R) 和绿色 (G) 通道为条件，绿色通道 (G) 以红色 (R) 通道为条件。 掩码机制会分裂特征图，并调整掩码张量的中心值，以确保在预测某个颜色通道时，只能访问当前像素中已经被预测的通道。
即，作者**假设RGB三个通道之间存在相互影响**：
- 其中**红色预测**不受蓝色和绿色通道的影响，**只受上下文影响**；
- **绿色**受**红色**通道和**上下文影响**，但不受蓝色通道影响；
- **蓝色**通道受**上下文、红色通道、绿色通道影响**。

#### 采样
在PixelCNN的图像生成过程中，需要通过采样将模型预测的​**​概率分布转换为具体的像素值​**​，并执行归一化操作。
1. **模型预测​**​：模型 `output` 给出了当前像素 `(i, j)` 所有可能颜色值的原始预测（logits）。
2. ​**​概率转换​**​：通过 softmax 将这些 logits 变成概率分布 `prob_dist`。
3. ​**​采样​**​：从概率分布中随机抽取一个颜色值（确保生成多样性）。
4. ​**​归一化​**​：将离散索引映射到[0, 1]区间，并更新图像张量 `x` 的当前位置。
5. ​**​迭代生成​**​：循环遍历所有 `(i, j)` 位置，每次用新像素更新 `x`，逐步生成完整图像。
- **采样（`multinomial`）​**​：**避免生成结果确定性**（即总生成相同的图像），引入随机性使每次生成结果不同。
```python
def sample(model, device, model_path, output_path, n_sample=1):
    """
        把x初始化成一个0张量。
        循环遍历每一个像素，输入x，把预测出的下一个像素填入x
    """
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    C, H, W = get_img_shape()  # (1, 28, 28)
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                # 我们先获取模型的输出，再用softmax转换成概率分布
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                # 再用torch.multinomial从概率分布里采样出【1】个[0, color_level-1]的离散颜色值
                # 再除以(color_level - 1)把离散颜色转换成浮点[0, 1]
                pixel = torch.multinomial(input=prob_dist, num_samples=1).float() / (color_level - 1)
                # 最后把新像素填入到生成图像中
                x[:, :, i, j] = pixel
    # 乘255变成一个用8位字节表示的图像
    imgs = x * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite(output_path, imgs)
```

#### Gated PixelCNN
原始的 PixelCNN 的感受野随着层数的增加线性增长，并且存在一个“**盲点**”，即无法考虑当前像素右侧的信息来进行预测。为了弥补 PixelCNN 的这些不足并提高性能，提出了 门控 PixelCNN 即Gated PixelCNN。Gated PixelCNN 引入了门控激活单元和双堆栈结构（垂直堆栈和水平堆栈）来解决感受野中的盲点问题。这些改进使得 Gated PixelCNN 在 ImageNet 数据集上能够匹敌甚至超越 PixelRNN 的性能，同时保持了计算效率的优势。

**门控运算的通用实现**
门控机制的核心是​**​信息筛选操作​**​，公式为：
```python
# 公式
# output = gating_function(context) ⊙ transformation_function(content)

# Gated PixelCNN 实现
def GatedActivation(self, x):
	return torch.tanh(x[:, :self.nfeats]) * torch.sigmoid(x[:, self.nfeats:])
```

**门控单元:** LSTM中则通过**可学习的参数**来决定信息的保留与遗忘。
```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 组合参数矩阵 (输入门, 遗忘门, 候选记忆, 输出门)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, state):
        # state = (hidden_state, cell_state)
        h, c = state
        
        # 所有门一起计算
        gates = (x @ self.weight_ih.T + self.bias) + (h @ self.weight_hh.T)
        
        # 分割出各个门
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, 1)
        
        # 应用非线性激活函数
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        
        # 更新细胞状态
        new_c = forget_gate * c + input_gate * candidate
        
        # 计算新隐藏状态
        new_h = output_gate * torch.tanh(new_c)
        
        return new_h, new_c
```


### Reference

参考链接：[经典神经网络(10)PixelCNN模型、Gated PixelCNN模型及其在MNIST数据集上的应用-CSDN博客](https://blog.csdn.net/qq_44665283/article/details/139533111)
