import torch


class Config:
    # 联邦学习配置
    NUM_CLIENTS = 5 #总客户端数量
    NUM_ROUNDS = 100    #联邦训练总轮数
    FRACTION = 0.5  # 每轮选择的客户端比例

    # 模型配置
    SERVER_LAYERS = 2  # 客户端后n层的层数，即共享层的层数
    CLIENT_LAYERS = [5, 4, 3 , 4 , 5 ]  # 各客户端的m_i值，即各个客户端神经网络的层数

    # 训练配置
    BATCH_SIZE = 64 #每个DataLoader的batch大小
    CLIENT_EPOCHS = 3   #客户端训练私有层/共享层时的本地epoch次数（在代码中用于私有层训练）
    LEARNING_RATE = 0.01

    # 数据配置
    INPUT_SIZE = 784  # MNIST图像大小为28*28 = 784
    HIDDEN_SIZE = 128   #隐藏层的宽度
    NUM_CLASSES = 10    #分类类别数量

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")