import numpy as np


class Dimensions:

    def __init__(self) -> None:
        self.types = {
            'conv': self._dims_for_conv,
            'pool': self._dims_for_pool
        }

    def _function_for_layer(self, layer):
        functions = [self.types[type_str] for type_str in self.types.keys()
                     if type_str in layer.__class__.__name__.lower()]
        return functions[0] if len(functions) > 0 else None

    def _dims_for_layer(self, img_dims, layer):
        dims_function = self._function_for_layer(layer)
        return dims_function(img_dims, layer)

    def dimensions_for_layers(self, img_dims, layers):

        if len(layers) == 1:
            return self._dims_for_layer(img_dims, layers)
        else:
            output_dims = [img_dims]
            for layer in layers:
                output_dims.append(self._dims_for_layer(output_dims[-1], layer))
            return output_dims

    def dimensions_for_linear(self, img_dims, layers):
        output_dims = self.dimensions_for_layers(img_dims, layers)
        return np.product(output_dims[-1])

    @staticmethod
    def _calc_dims(img_dims, kernel_size, stride, padding, dilation):
        # from subsection 'Shape' in https://pytorch.org/docs/stable/nn.html#conv3d
        # from subsection 'Shape' in  https://pytorch.org/docs/stable/nn.html#maxpool3d
        output_dims = ((img_dims + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        return output_dims

    def _calc_dims_for_layer(self, img_dims, layer):
        kernel_size = np.squeeze(np.array(layer.kernel_size))
        stride = np.squeeze(np.array(layer.stride))
        dilation = np.squeeze(np.array(layer.dilation)) if hasattr(layer, 'dilation') else 0
        padding = np.squeeze(np.array(layer.padding)) if hasattr(layer, 'padding') else 0
        return self._calc_dims(img_dims, kernel_size, stride, padding, dilation)

    def _dims_for_pool(self, img_dims, layer):
        img_dims = np.array(img_dims)
        # output_img_dims = ((img_dims[1:] - dilation * (kernel_size - 1) - 1) / stride) + 1
        output_img_dims = self._calc_dims_for_layer(img_dims[1:], layer)
        output_dims = [img_dims[0]]
        output_dims.extend(output_img_dims.astype(np.int).tolist())

        return output_dims

    def _dims_for_conv(self, img_dims, layer):
        img_dims = np.array(img_dims)
        # output_img_dims = (img_dims[1:] - kernel_size + stride) / stride
        output_img_dims = self._calc_dims_for_layer(img_dims[1:], layer)
        output_dims = [layer.out_channels]
        output_dims.extend(output_img_dims.astype(np.int).tolist())

        return tuple(output_dims)
