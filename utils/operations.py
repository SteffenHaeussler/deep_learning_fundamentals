import numpy as np
from numpy import ndarray

from .np_utils import assert_same_shape


class Operation:
    """Base class for an artificial neural network"""

    def __init__(self):
        pass

    def forward(self, input_: ndarray) -> ndarray:
        """
        stores input in the self.input_ instance variable
        calls """
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        calls the _input_grad function
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self) -> ndarray:

        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()


class ParamOperation(Operation):
    """
    allows operation with parameters
    """

    def __init__(self, param: ndarray) -> ndarray:

        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input and self._param_grad
        """

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    """Weight multiplication for a neural network"""

    def __init__(self, W: ndarray):
        """Init Operation with self.param = W"""
        super().__init__(W)

    def _output(self) -> ndarray:

        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """Add bias"""

    def __init__(self, B: ndarray):
        """Init Operation with self.param = B"""
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:

        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    """Sigmoid activation function"""

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        sigmoid_backward = self.output * (1 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    Identity" activation function
    """

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> ndarray:

        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad


class Tanh(Operation):
    """
    Identity" activation function
    """

    def __init__(self) -> None:

        super().__init__()

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - np.power(self.output, 2))
