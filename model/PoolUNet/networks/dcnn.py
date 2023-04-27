import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1

    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""
    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])

    return a


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)
    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()

    return Variable(grid, requires_grad=False)


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates"""

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)
    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
                        torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input"""
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)
    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)
    coords = offsets + grid
    mapped_vals = th_batch_map_coordinates(input, coords)  # B, C, H

    return mapped_vals


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, in_channels, init_normal_stddev=0.01, **kwargs):
        self.filters = in_channels
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters * 2, (3, 3), padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        x_shape = x.size()
        offsets_ = super(ConvOffset2D, self).forward(x)
        offsets = self._to_bc_h_w_2(offsets_, x_shape)  # B*C, H, W, 2
        x = self._to_bc_h_w(x, x_shape)  # B*C, H, W
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))  # B*C, H, W
        x_offset = self._to_b_c_h_w(x_offset, x_shape)  # B, C, H, W

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x
