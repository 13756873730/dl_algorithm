import torch

"""
    Numpy实现两层神经网络
    h = w1 * x
    a = max(0, h)
    y_hat = w2 * x
    
    input_dim=1000 ---w1--> hidden_dim=100 ---w2--> output_dim=10
    
"""
if __name__ == '__main__':
    input_count = 64
    input_dim = 1000
    hidden_dim = 100
    output_dim = 10

    X = torch.randn(input_count, input_dim)
    y = torch.randn(input_count, output_dim)

    nn_model = torch.nn.Sequential(
        # 默认bias=True
        torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
    )

    # 手动初始化模型参数
    # torch.nn.init.normal_(nn_model[0].weight)
    # torch.nn.init.normal_(nn_model[2].weight)

    # 查看模型结构
    print(nn_model)
    for layer in nn_model:
        print(layer)

    # 定义损失函数
    loss_function = torch.nn.MSELoss(reduction='mean')

    # 定义最优化方法
    learning_rate = 1e-4
    # optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(params=nn_model.parameters(), lr=learning_rate)

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
