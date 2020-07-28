import numpy as np
from numpy import ndarray

from .np_utils import assert_same_shape


class Optimizer:
    """
    Base class for an optimizer.
    """

    def __init__(self, lr: float = 0.01, final_lr: float = 0.0, decay_type: str = None):

        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:

        if not self.decay_type:
            return None

        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr / self.lr, 1.0 / (self.max_epochs - 1)
            )

    def _decay_lr(self) -> None:

        if not self.decay_type:
            return None

        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

        elif self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch

    def step(self, epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):

            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    """
    Stochasitc gradient descent optimizer.
    """

    def __init__(self, lr: float = 0.01) -> None:

        super().__init__(lr)

    def step(self):
        """
        Update each parameter ased on the learning rate.
        """
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):

            param -= self.lr * param_grad


class SGDMomentum(Optimizer):
    """
    Stochasitc gradient descent optimizer with momentum.
    """

    def __init__(
        self,
        lr: float = 0.01,
        final_lr: float = 0,
        decay_type: str = None,
        momentum=0.9,
    ) -> None:

        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self):
        """
        Update each parameter ased on the learning rate.
        """
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(
            self.net.params(), self.net.param_grads(), self.velocities
        ):

            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

        # Update velocity
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]

        # Use this to update parameters
        kwargs["param"] -= kwargs["velocity"]


class Loss:
    """loss calculation of the network"""

    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """
        computes the actual loss value
        """
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        """
        computes gradient of the loss value with respect to the input of the loss function
        """
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.target)

        return self.input_grad

    def _output(self) -> float:
        """
        every subclass of "loss" had to implement the output function!
        """
        raise NotImplementedError()

    def _input_grad(self) -> float:
        """
        every subclass of "loss" had to implement the input_grad function!
        """
        raise NotImplementedError()


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        """
        computes the observation squared error loss
        """
        loss = (
            np.sum(np.power(self.prediction - self.target, 2))
            / self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:
        """
        calculates the loss gradient with respect to the input of the mse loss
        """
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


# TODO: proper implementation - this is copy/paste
class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-9) -> None:

        super().__init__()
        self.eps = eps

    def _output(self) -> float:

        if self.target.shape[1] == 0:
            raise NotImplementedError()

        softmax_preds = softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
            1.0 - self.target
        ) * np.log(1 - self.softmax_preds)

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:

        return (self.softmax_preds - self.target) / self.prediction.shape[0]
