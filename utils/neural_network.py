from typing import List, Tuple

import numpy as np
from numpy import ndarray

from .np_utils import assert_same_shape, permute_data
from .layers import Layer
from .optimizers import Loss, Optimizer
from .operations import WeightMultiply


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):

        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        """
        passes data forward through the series of layers
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        """
        passes loss gradient backward through the series of layers
        """

        grad = loss_grad

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        """

        prediction = self.forward(x_batch)

        loss = self.loss.forward(prediction, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        """
        Gets parameters for the network.
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        Gets the gradient of the loss with respect to the parameters for the network.
        """
        for layer in self.layers:
            yield from layer.param_grads


class Trainer:
    """
    Trains a neural network
    """

    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        """
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.history = []
        setattr(self.optim, "net", self.net)

    def generate_batches(
        self, X: ndarray, y: ndarray, size: int = 32
    ) -> Tuple[ndarray]:
        """
        Generates training batches
        """
        assert (
            X.shape[0] == y.shape[0]
        ), """
            features and target must have the same number of rows, instead
            features has {0} and target has {1}
            """.format(
            X.shape[0], y.shape[0]
        )

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch

    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_valid: ndarray,
        y_valid: ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
        record_history: bool = False,
    ) -> None:
        """
        Fits the neural network on the training data
        Every "eval_every" epochs, it evaluated the neural network on the validation data.
        """
        setattr(self.optim, 'max_epochs', epochs)
        self.optim._setup_decay()

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

        for e in range(epochs):

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if record_history:

                valid_preds = self.net.forward(X_valid)
                val_loss = self.net.loss.forward(valid_preds, y_valid)

                train_preds = self.net.forward(X_train)
                train_loss = self.net.loss.forward(train_preds, y_train)

                self.history.append({"val_loss": val_loss,
                                     "train_loss": train_loss})


            if (e + 1) % eval_every == 0:

                valid_preds = self.net.forward(X_valid)
                loss = self.net.loss.forward(valid_preds, y_valid)

                print(f"Validation loss after {e+1} epochs is {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()
