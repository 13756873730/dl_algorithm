import torch


class TwoLayerModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerModel, self).__init__()
        # Model Architecture
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # Forward Propagation
        y_predict = self.layer2(self.layer1(X).clamp(min=0))
        return y_predict


if __name__ == '__main__':
    input_count = 64
    input_dim = 1000
    hidden_dim = 100
    output_dim = 10

    X = torch.randn(input_count, input_dim)
    y = torch.randn(input_count, output_dim)

    nn_model = TwoLayerModel(input_dim, hidden_dim, output_dim)

    # 查看模型结构
    print(nn_model)

    # 定义损失函数
    loss_function = torch.nn.MSELoss(reduction='mean')

    # 定义最优化方法
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=learning_rate)

    for index in range(2000):
        # Forward Propagation
        y_predict = nn_model(X)

        # Loss Function
        loss = loss_function(y_predict, y)  # loss=MSE
        if index % 100 == 0:
            print(index, loss.item())

        # Back Propagation
        # torch可以自动求解梯度，requires_grad默认为False
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    pass
