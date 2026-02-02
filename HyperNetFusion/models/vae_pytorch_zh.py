import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 定义VAE模型类
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dim=512, activation=nn.ReLU):
        super(VAE, self).__init__()  # 初始化父类（nn.Module）
        self.latent_dim = latent_dim  # 设置潜在空间的维度

        # Encoder: 将输入数据映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入维度到隐藏层维度的线性变换
            activation(),  # 激活函数（默认ReLU）
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和对数方差（latent_dim * 2是因为分别输出均值和对数方差）
        )

        # Decoder: 从潜在空间重构输入数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # 从潜在空间到隐藏层的线性变换
            activation(),  # 激活函数（默认ReLU）
            nn.Linear(hidden_dim, input_dim),  # 从隐藏层到输入维度的线性变换
            nn.Sigmoid()  # 使用Sigmoid激活函数确保输出值在0到1之间
        )

    def encode(self, x):
        # 编码：将输入x通过编码器得到均值和对数方差
        params = self.encoder(x)
        z_mean, z_log_var = params[:, :self.latent_dim], params[:, self.latent_dim:]  # 分割均值和对数方差
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        # 重参数化技巧：通过均值和对数方差从正态分布采样
        std = torch.exp(0.5 * z_log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 随机噪声
        return z_mean + eps * std  # 返回采样结果

    def decode(self, z):
        # 解码：从潜在空间z重构数据
        return self.decoder(z)

    def forward(self, x):
        # 前向传播
        z_mean, z_log_var = self.encode(x)  # 编码输入
        z = self.reparameterize(z_mean, z_log_var)  # 重参数化采样潜在变量
        return self.decode(z), z_mean, z_log_var  # 返回重构的数据和潜在变量的均值与对数方差
class VAEDataAugmentationPyTorch:
    def __init__(self, input_dim, latent_dim=20, hidden_dim=512, learning_rate=1e-3, kl_weight=1.0, recon_weight=1.0,
                 l1_reg=0.0, l2_reg=0.0, activation=nn.ReLU, early_stopping=True, patience=10,
                 num_interpolation_points=5, device='cpu'):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activation = activation  # 激活函数
        self.early_stopping = early_stopping  # 是否启用早停
        self.patience = patience  # 早停的耐心值
        self.num_interpolation_points = num_interpolation_points  # 插值点的数量
        self.device = device  # 设备（CPU或GPU）

        self.model = VAE(input_dim, latent_dim, hidden_dim, activation).to(self.device)  # 初始化VAE模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam优化器

    def loss_function(self, recon_x, x, z_mean, z_log_var):
        # 计算VAE的损失函数（重构损失和KL散度）
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')  # 二进制交叉熵损失
        KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())  # KL散度
        return self.recon_weight * BCE + self.kl_weight * KLD  # 总损失（重构损失和KL散度的加权和）

    def train_model(self, dataloader, epochs=50):
        # 训练VAE模型
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)  # 将输入数据移到指定设备
                self.optimizer.zero_grad()  # 清除梯度
                recon_batch, z_mean, z_log_var = self.model(batch_x)  # 前向传播
                loss = self.loss_function(recon_batch, batch_x, z_mean, z_log_var)  # 计算损失
                loss.backward()  # 反向传播
                train_loss += loss.item()  # 累加损失
                self.optimizer.step()  # 更新参数
            print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset):.4f}')  # 打印训练损失

    def encode(self, x):
        # 编码：通过VAE模型得到潜在空间的均值
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            z_mean, z_log_var = self.model.encode(x.to(self.device))  # 得到均值和对数方差
            return z_mean.cpu().numpy()  # 返回均值

    def decode(self, z):
        # 解码：将潜在空间的点转换为重构数据
        self.model.eval()
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32).to(self.device)  # 将潜在变量转为tensor
            recon_x = self.model.decode(z)  # 解码
            return recon_x.cpu().numpy()  # 返回重构的数据

    def augment_data(self, X, y):
        # 基于类别插值进行数据增强
        unique_classes = np.unique(y)  # 获取所有唯一类别
        scaler = MinMaxScaler()  # 数据归一化器
        x_scaled = scaler.fit_transform(X)  # 对数据进行归一化

        augmented_data = []  # 存储增强后的数据
        augmented_labels = []  # 存储增强后的标签

        for cls in unique_classes:
            x_class = x_scaled[y == cls]  # 获取属于该类别的数据
            dataset = CustomDataset(x_class)  # 创建自定义数据集
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # 创建数据加载器
            self.train_model(dataloader, epochs=50)  # 训练VAE模型

            z_mean = self.encode(dataset.data)  # 得到潜在空间的均值
            recon_x = self.decode(z_mean)  # 解码得到重构数据

            # 在原始数据和重构数据之间进行插值
            for original, decoded in zip(x_class, recon_x):
                interpolated_points = linear_interpolate_points(original, decoded, self.num_interpolation_points)  # 插值
                augmented_data.extend(interpolated_points[1:-1])  # 避免使用原始点和重构点
                augmented_labels.extend([cls] * self.num_interpolation_points)  # 为每个插值点添加标签

        augmented_data = np.array(augmented_data)  # 转换为数组
        augmented_labels = np.array(augmented_labels)  # 转换为数组

        # 逆归一化，恢复到原始数据范围
        augmented_data_inverse = scaler.inverse_transform(augmented_data)
        return augmented_data_inverse, augmented_labels  # 返回增强后的数据和标签
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # 将数据转换为tensor

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.data[idx], 0  # 返回数据和一个虚拟标签
def linear_interpolate_points(point_a, point_b, num_points=5):
    # 线性插值：在两个点之间生成新样本
    return np.linspace(point_a, point_b, num=num_points + 2)  # 包括原始点和解码点
