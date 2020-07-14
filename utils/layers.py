from typing import List

import numpy as np
from numpy import ndarray

from .np_utils import assert_same_shape
from .operations import Operation, Sigmoid
from .operations import ParamOperation, BiasAdd, WeightMultiply


class Layer:
    """layer in a neural network"""

    def __init__(self, neurons: int):

        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grad: List[ndarray] = []
        self.operations: List[ndarray] = []

    def _setup_layer(self, num_in: int) -> None:

        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:

        self.param_grads = []

        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """fully connected layer"""

    def __init__(self, neurons: int, activation: Operation = Sigmoid()) -> None:
        """requires an activation function upon initialization"""

        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        """
        defines options for a fully connected layer
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]

        return None
