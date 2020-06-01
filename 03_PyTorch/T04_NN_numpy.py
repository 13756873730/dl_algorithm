import numpy as np

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

    X = np.random.randn(input_count, input_dim)
    y = np.random.randn(input_count, output_dim)

    w1 = np.random.randn(input_dim, hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim)

    learning_rate = 1e-6
    for index in range(500):
        # Forward Propagation
        h = X.dot(w1)  # 隐藏层
        h_relu = np.maximum(0, h)  # activation='relu'
        y_predict = h_relu.dot(w2)  # 输出层

        # Loss Function
        loss = 1.0 / input_count * np.sum(np.power((y_predict - y), 2.0))  # loss=MSE

        if index % 20 == 0:
            print(index, loss)

        # Back Propagation
        grad_y_predict = 2.0 * (y_predict - y)
        grad_w2 = h_relu.T.dot(grad_y_predict)
        grad_h_relu = grad_y_predict.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = X.T.dot(grad_h)

        # 梯度下降
        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2
    pass
