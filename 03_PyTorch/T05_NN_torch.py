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

    w1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
    w2 = torch.randn(hidden_dim, output_dim, requires_grad=True)

    learning_rate = 1e-4
    for index in range(3000):
        # Forward Propagation
        # 简化写法
        y_predict = X.mm(w1).clamp(min=0).mm(w2)

        # Loss Function
        loss = 1.0 / input_count * (y_predict - y).pow(2.0).sum()  # loss=MSE
        if index % 100 == 0:
            print(index, loss.item())

        # Back Propagation
        # torch可以自动求解梯度，requires_grad默认为False
        loss.backward()

        # 梯度下降
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
    pass
