import torch
import numpy

if __name__ == '__main__':
    x = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])
    print('x:')
    print(x)
    print('------------------------------------------------')

    # x.data
    print('x.data:')
    print(x.data)
    print('------------------------------------------------')

    # x.grad
    print('x.grad:')
    print(x.grad)
    print('------------------------------------------------')

    # tensor和ndarray互转
    x_numpy = x.numpy()
    print(type(x_numpy))

    x_tensor = torch.from_numpy(x_numpy)
    print(type(x_tensor))
    print('------------------------------------------------')

    # torch tensor 和 numpy ndarray共享内存
    x_numpy = x.numpy()
    print('x:')
    print(x)
    print('x_numpy:', )
    print(x_numpy)

    x[0, 0] = 100.0

    print('x:')
    print(x)
    print('x_numpy:', )
    print(x_numpy)
    print('------------------------------------------------')

    print(torch.cuda.is_available())
    print('------------------------------------------------')
    pass
