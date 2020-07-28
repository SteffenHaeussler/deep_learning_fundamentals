import numpy as np
from numpy import ndarray

from .operations import ParamOperation


class Conv2D_Op(ParamOperation):

    def __init__(self, weights):
        super().__init__(weights)

        self.param_size = weights.shape[2]
        # simplification for training purpose
        self.param_pad = self.param_size // 2

    def _pad_1d(self, input_: ndarray) -> ndarray:
        """
        pads zeros around the input based on the padding size
        """
        zeros = np.repeat(np.array([0]), self.param_pad)
        return np.concatenate([zeros, input_, zeros])

    def _pad_1d_batch(self, input_: ndarray) -> ndarray:

        output = [self._pad_1d(obs) for obs in input_]

        return np.stack(output)

    def _pad_2d_obs(self, input_: ndarray):

        input_pad = self._pad_1d_batch(input_)

        other = np.zeros((self.param_pad, input_.shape[0] + self.param_pad * 2))

        return np.concatenate([other, input_pad, other])

    def _pad_2d_channel(self, input_: ndarray):

        return np.stack([self._pad_2d_obs(channel) for channel in input_])

    def _get_image_patches(self, input_: ndarray):

        patches = []

        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])

        img_height = imgs_batch_pad.shape[2]

        for h in range(img_height-self.param_size+1):
            for w in range(img_height-self.param_size+1):

                patch = imgs_batch_pad[:, :, h:h+self.param_size, w:w+self.param_size]
                patches.append(patch)

        return np.stack(patches)

    def _output(self) -> ndarray:

        batch_size = self.input_.shape[0]

        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        #step 1 and 2
        output_patches_reshaped = (self._get_image_patches(self.input_)
                                  .transpose(1, 0, 2, 3, 4)
                                  .reshape(batch_size, img_size, -1))

        # step 3
        param_reshaped = (self.param
                          .transpose(0, 2, 3, 1)
                          .reshape(patch_size, -1))

        # step 4 and step 5
        return (np.matmul(output_patches_reshaped, param_reshaped)
                .reshape(batch_size, img_height, img_height, -1)
                .transpose(0, 3, 1, 2))

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        batch_size = self.input_.shape[0]

        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (self._get_image_patches(output_grad)
                          .transpose(1, 0, 2, 3, 4)
                          .reshape(batch_size * img_size, -1))

        param_reshaped = (self.param
                          .reshape(self.param.shape[0], -1)
                          .transpose(1, 0))

        return (np.matmul(output_patches, param_reshaped)
                .reshape(batch_size, img_height, img_height, self.param.shape[0])
                .transpose(0, 3, 1, 2))

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        batch_size = self.input_.shape[0]

        img_size = self.input_.shape[2] * self.input_.shape[3]

        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        #step 1 and step 2 and step 3
        in_patches_reshape = (self._get_image_patches(self.input_)
                                        .reshape(batch_size * img_size, -1)
                                        .transpose(1,0))

        # transpose to match general pattern for backward gradient
        out_grad_reshape = output_grad.transpose(0,2,3,1).reshape(batch_size * img_size, -1)

        # step 4 and step 5
        return (np.matmul(in_patches_reshape, out_grad_reshape)
                        .reshape(in_channels, self.param_size, self.param_size, out_channels)
                        .transpose(0,3,1,2))
