# 导入相关的库
import pandas as pd  # 用于数据处理的库
import numpy as np  # 用于科学计算的库
from torch import nn  # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # PyTorch数据集和数据加载器
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 用于数据标准化和标签编码的库
from models.vae_pytorch import VAEDataAugmentationPyTorch  # 引入自定义的VAE数据增强类
import torch  # PyTorch的核心库
import sys  # 系统操作模块

# 定义自定义的TabularDataset类，继承自Dataset类，方便PyTorch加载数据
class TabularDataset(Dataset):
    def __init__(self, data, labels, scaler=None):
        self.labels = labels  # 目标标签
        if scaler:  # 如果传入了预处理器（如StandardScaler）
            self.scaler = scaler
            self.data = self.scaler.transform(data)  # 对数据进行标准化处理
        else:
            self.scaler = StandardScaler()  # 如果没有传入预处理器，创建一个新的StandardScaler
            self.data = self.scaler.fit_transform(data)  # 对数据进行标准化处理并计算均值和标准差

    def __len__(self):
        return len(self.labels)  # 返回数据集的长度，即样本数

    def __getitem__(self, idx):
        # 返回第idx个样本的数据和标签，将它们转换为PyTorch张量
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# 定义一个函数来加载训练和验证数据，并进行必要的数据预处理和增强
def get_data_loaders(csv_path, target_column, train_indices, val_indices, batch_size=32, augment=False):
    df = pd.read_csv(csv_path)  # 读取CSV文件为DataFrame
    df.columns = df.columns.str.strip().str.lower()  # 去除列名中的空格并将列名转换为小写
    print(f"Processed column names: {df.columns.tolist()}")  # 打印处理后的列名

    if target_column.lower() not in df.columns:  # 检查目标列是否存在
        print(f"Error: Target column '{target_column}' does not exist in the CSV file.")
        sys.exit(1)  # 如果目标列不存在，则退出程序

    # 去除目标列和"sample id"列，选取特征列
    feature_columns = [col for col in df.columns if col not in [target_column.lower(), 'sample id']]
    X = df[feature_columns].values  # 提取特征数据
    y = df[target_column.lower()].values  # 提取目标标签数据

    # 使用LabelEncoder将字符串标签转换为整数标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # 对标签进行编码
    print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # 根据传入的索引切分训练集和验证集
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y_encoded[train_indices], y_encoded[val_indices]

    # 如果启用数据增强，则进行数据增强（仅对训练集进行）
    if augment:
        augmenter = VAEDataAugmentationPyTorch(
            input_dim=X_train.shape[1],  # 输入特征的维度
            latent_dim=20,  # 潜在空间的维度
            hidden_dim=512,  # 隐藏层维度
            learning_rate=0.001,  # 学习率
            kl_weight=1.0,  # KL散度的权重
            recon_weight=1.0,  # 重构误差的权重
            l1_reg=0.0,  # L1正则化的权重
            l2_reg=0.0,  # L2正则化的权重
            activation=nn.ReLU,  # 激活函数类型
            early_stopping=True,  # 是否启用早停
            patience=10,  # 早停耐心值
            num_interpolation_points=5,  # 插值点的数量
            device='cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择（CUDA或CPU）
        )
        # 执行数据增强
        augmented_X, augmented_y = augmenter.augment_data(X_train, y_train)
        # 将增强后的数据与原始训练数据合并
        X_train = np.vstack((X_train, augmented_X))
        y_train = np.concatenate((y_train, augmented_y))
        print(
            f"Original training data size: {len(y_train) - len(augmented_y)}, Augmented data size: {len(augmented_y)}, Total training data size: {len(y_train)}"
        )

    # 仅对训练数据进行标准化
    scaler = StandardScaler()
    scaler.fit(X_train)  # 在训练数据上拟合标准化器

    # 创建训练集和验证集的数据集对象
    train_dataset = TabularDataset(X_train, y_train, scaler=scaler)
    val_dataset = TabularDataset(X_val, y_val, scaler=scaler)  # 验证集使用相同的标准化器

    # 使用DataLoader将数据集封装为可以批量加载的形式
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集使用洗牌
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集不洗牌

    return train_loader, val_loader, scaler, label_encoder  # 返回训练和验证数据加载器、标准化器和标签编码器
