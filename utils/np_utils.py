import numpy as np

from numpy import ndarray


def assert_same_shape(array: ndarray, array_grad: ndarray):

    assert (
        array.shape == array_grad.shape
    ), """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(
        tuple(array.shape), tuple(array_grad.shape)
    )

    return None


def to_2d_np(a: ndarray,
          type: str="col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

