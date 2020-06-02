import torch
import numpy

if __name__ == '__main__':
    tensor = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.]
    ])
    array = numpy.array([
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]
    ])

    # torch.tensor --> numpy.array
    print('----------- torch.tensor --> numpy.array ------------')
    tensor2array = tensor.numpy()
    print(type(tensor), type(tensor2array))
    print(tensor2array)
    print('-----------------------------------------------------')

    # numpy.array --> torch.tensor
    print('----------- numpy.array --> torch.tensor ------------')
    array2tensor = torch.from_numpy(array)
    print(type(array), type(array2tensor))
    print(array2tensor)
    print('-----------------------------------------------------')

    pass
