import torch

if __name__ == '__main__':
    x_tensor = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.]
    ])
    y_tensor = torch.tensor([
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]
    ])
    n = 2.

    # __add__
    print('----------- __add__ ------------------')
    z_tensor = x_tensor + y_tensor
    print(z_tensor)
    print('--------------------------------------')

    # __sub__
    print('----------- __sub__ ------------------')
    z_tensor = x_tensor - y_tensor
    print(z_tensor)
    print('--------------------------------------')

    # __mul__
    print('----------- __mul__ ------------------')
    z_tensor = x_tensor * n
    print(z_tensor)
    print('--------------------------------------')

    # __div__
    print('----------- __div__ ------------------')
    z_tensor = x_tensor / n
    print(z_tensor)
    print('--------------------------------------')

    # 支持切片
    print('----------- 支持切片 -----------------')
    z_tensor = x_tensor[:2, 1:]  # 0~1, 1~3
    print(z_tensor)
    print('--------------------------------------')

    pass
