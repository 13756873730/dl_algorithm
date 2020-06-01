import torch

if __name__ == '__main__':
    # 空tensor
    tensor = torch.empty(size=[3, 5])
    print(type(tensor))  # <class 'torch.Tensor'>
    print(tensor)
    print('------------------------------------------------')

    # 随机tensor (0, 1)
    tensor = torch.rand(size=[3, 5])
    print(tensor)
    print('------------------------------------------------')

    # 全零
    tensor = torch.zeros(size=[3, 5], dtype=torch.int)
    print(tensor)
    print(tensor.dtype)
    print('------------------------------------------------')

    # 直接torch.tensor(data=list)创建，类似np.array(list)
    tensor = torch.tensor(data=[1.0, 2., 3])
    print(tensor)
    print(tensor.shape)
    print('------------------------------------------------')

    # 产生shape相同的随机tensor
    tensor = torch.zeros(size=[3, 5])
    print(tensor)
    tensor = torch.randn_like(tensor)
    print(tensor)
    print('------------------------------------------------')

    # shape属性 size()方法作用相同
    print(tensor.shape)
    print(tensor.size())
    print('------------------------------------------------')

    pass
