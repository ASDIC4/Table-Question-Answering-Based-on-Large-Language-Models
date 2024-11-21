from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == "__main__":

    print(torch.__version__)

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())

    # # 将模型移动到GPU(如果有GPU的话)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 定义优化器和学习率调度器
    # model = Model(
    #     alpha_1 = 1.0, alpha_2 = 1.0,
    #     theta_1 = 0.06, theta_2 = 0.06, theta_3 = 0.06, gamma = 10.0
    # ).to(device)
    # print("Device",device)