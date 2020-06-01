import torch

if __name__ == '__main__':
    x = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.]
    ])
    y = torch.tensor([
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]
    ])
    print(x.shape, y.shape)
    print('------------------------------------------------')

    # 加法
    print(x + y)
    # print(x.__add__(y))
    # print(torch.add(x, y))
    print('------------------------------------------------')

    # 减法
    print(x - y)
    # print(x.__sub__(y))
    # print(torch.sub(x, y))
    print('------------------------------------------------')

    # 支持切片
    print(x[:2, 1:])
    print('------------------------------------------------')

    # reshape
    print(x.shape)
    x = x.reshape([2, -1])
    print(x)
    print('------------------------------------------------')
    pass
