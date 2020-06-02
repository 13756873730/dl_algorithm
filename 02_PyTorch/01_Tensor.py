import torch

if __name__ == '__main__':
    # torch.empty(size, dtype): 空tensor，默认dtype=torch.float32，值为0.0
    print('----------- torch.empty ------------------')
    tensor = torch.empty(size=[3, 5])
    print(type(tensor))  # <class 'torch.Tensor'>
    print(tensor.dtype)
    print(tensor)
    print('------------------------------------------')

    # torch.rand(size): 随机产生元素在0~1之间的tensor
    print('----------- torch.rand -------------------')
    tensor = torch.rand(size=[3, 5])
    print(tensor)
    print('------------------------------------------')

    # torch.zeros(size, dtype): 产生全0的tensor，默认dtype=torch.float32
    print('----------- torch.zeros ------------------')
    tensor = torch.zeros(size=[3, 5], dtype=torch.int)
    print(tensor.dtype)
    print(tensor)
    print('------------------------------------------')

    # torch.ones(size, dtype): 产生全1的tensor，默认dtype=torch.float32
    print('----------- torch.ones -------------------')
    tensor = torch.ones(size=[3, 5], dtype=torch.int)
    print(tensor.dtype)
    print(tensor)
    print('------------------------------------------')

    # torch.zeros_like(tensor, dtype): 产生size、dtype相同的tensor，可指定dtype
    print('----------- torch.zeros_like -------------')
    tensor_like = torch.zeros_like(tensor, dtype=torch.float32)
    print(tensor_like.dtype)
    print(tensor_like)
    print('------------------------------------------')

    # torch.tensor(data): 直接使用列表创建tensor，相当于np.array([...])
    print('----------- torch.tensor -----------------')
    tensor = torch.tensor(data=[1., 2., 3., 4., 5., 6])
    print(tensor)
    print('------------------------------------------')

    # torch.reshape(tensor, shape)/obj.reshape(shape): 与np相同
    print('-------- torch.reshape/obj.reshape -------')
    tensor = torch.reshape(tensor, shape=[-1, 2])
    print(tensor)
    tensor = tensor.reshape(shape=[-1, 3])
    print(tensor)
    print('------------------------------------------')

    # obj.shape/obj.size(): shape属性和size()方法作用相同，均返回Size对象
    print('-------- obj.shape/obj.size() -------')
    tensor = torch.tensor(data=[1., 2., 3., 4., 5., 6])
    tensor = tensor.reshape(shape=[-1, 2])
    print(tensor.shape, tensor.size())
    print('------------------------------------------')

    pass
