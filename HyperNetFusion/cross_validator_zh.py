import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data.data_loader import get_data_loaders  # 用于加载数据
from models.hypernet_fusion import HyperNetFusion  # 引入HyperNetFusion模型
from utils.utils import save_model, calculate_accuracy  # 保存模型和计算准确率的函数

class CrossValidator:
    def __init__(self, csv_path, target_column, num_folds=5, augment=False, batch_size=32, num_epochs=100, learning_rate=1e-3):
        self.csv_path = csv_path  # 数据集路径
        self.target_column = target_column  # 目标列名
        self.num_folds = num_folds  # 交叉验证的折数
        self.augment = augment  # 是否进行数据增强
        self.batch_size = batch_size  # 批量大小
        self.num_epochs = num_epochs  # 训练的周期数
        self.learning_rate = learning_rate  # 学习率

        # 超网络和目标网络的超参数
        self.hypernet_input_dim = 4  # 超网络的输入维度
        self.hypernet_hidden_dim = 128  # 超网络的隐藏层维度
        self.target_net_input_dim = 4  # 目标网络输入的特征维度
        self.target_net_hidden_dim = 64  # 目标网络的隐藏层维度
        self.target_net_output_dim = 2  # 目标网络的输出维度，二分类问题
        self.num_target_nets = 10  # 目标网络的数量

        self.results = []  # 存储每折的结果

    def run(self):
        # 读取整个数据集
        df = pd.read_csv(self.csv_path)  # 读取CSV文件
        df.columns = df.columns.str.strip().str.lower()  # 去除列名的空格并转为小写
        print(f"Processed column names: {df.columns.tolist()}")  # 打印处理后的列名

        if self.target_column.lower() not in df.columns:  # 检查目标列是否存在
            print(f"Error: Target column '{self.target_column}' not found in CSV file.")
            sys.exit(1)

        # 移除目标列和'sample id'列，提取特征数据
        feature_columns = [col for col in df.columns if col not in [self.target_column.lower(), 'sample id']]
        X = df[feature_columns].values
        y = df[self.target_column.lower()].values

        # 使用LabelEncoder将标签从字符串转换为整数
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        # 初始化分层K折交叉验证
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # 记录每一折的性能
        self.results = []

        for fold, (train_indices, val_indices) in enumerate(skf.split(X, y_encoded), 1):
            print(f"\n--- Fold {fold}/{self.num_folds} ---")
            # 获取当前折的训练和验证数据加载器
            train_loader, val_loader, scaler, _ = get_data_loaders(
                self.csv_path, self.target_column, train_indices, val_indices, batch_size=self.batch_size, augment=self.augment
            )
            print(f"Fold {fold}: Training size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")

            # 初始化模型
            print("Initializing model...")
            model = HyperNetFusion(
                hypernet_input_dim=self.hypernet_input_dim,
                hypernet_hidden_dim=self.hypernet_hidden_dim,
                target_net_input_dim=self.target_net_input_dim,
                target_net_hidden_dim=self.target_net_hidden_dim,
                target_net_output_dim=self.target_net_output_dim,
                num_target_nets=self.num_target_nets
            )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU（如果有）
            model.to(device)  # 将模型移动到设备（CPU或GPU）
            print(f"Model initialized, using device: {device}")

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)  # Adam优化器

            # 训练循环
            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)  # 将输入数据移到设备
                    batch_y = batch_y.to(device)  # 将标签移到设备

                    hypernet_input = batch_x  # 超网络的输入（可以直接使用batch_x）

                    optimizer.zero_grad()  # 清空梯度
                    outputs = model(batch_x, hypernet_input)  # 获取模型输出
                    loss = criterion(outputs, batch_y)  # 计算损失
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新权重

                    running_loss += loss.item() * batch_x.size(0)
                    running_corrects += torch.sum(outputs.argmax(1) == batch_y)  # 计算正确预测数

                # 输出每个epoch的损失和准确度
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_acc = running_corrects.double() / len(train_loader.dataset)
                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # 验证阶段
            model.eval()  # 设置为评估模式
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():  # 不计算梯度
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    hypernet_input = batch_x  # 超网络的输入（可以直接使用batch_x）

                    outputs = model(batch_x, hypernet_input)  # 获取输出
                    loss = criterion(outputs, batch_y)  # 计算损失

                    val_loss += loss.item() * batch_x.size(0)
                    val_corrects += torch.sum(outputs.argmax(1) == batch_y)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
            print(f'Fold {fold} Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')

            # 记录当前折的结果
            self.results.append({
                'fold': fold,
                'validation_loss': val_epoch_loss,
                'validation_accuracy': val_epoch_acc.item()
            })

        # 汇总所有折的结果
        avg_val_loss = np.mean([result['validation_loss'] for result in self.results])
        avg_val_acc = np.mean([result['validation_accuracy'] for result in self.results])
        std_val_loss = np.std([result['validation_loss'] for result in self.results])
        std_val_acc = np.std([result['validation_accuracy'] for result in self.results])

        print("\n=== Cross-validation Results ===")
        for result in self.results:
            print(f"Fold {result['fold']}: Validation Loss: {result['validation_loss']:.4f}, Validation Accuracy: {result['validation_accuracy']:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")

        return self.results, avg_val_loss, avg_val_acc, std_val_loss, std_val_acc
