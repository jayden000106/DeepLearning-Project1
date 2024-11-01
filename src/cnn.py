import numpy as np
from collections import OrderedDict

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride +1
    out_w = (W + 2*pad - filter_w) // stride +1

    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride +1
    out_w = (W + 2*pad - filter_w)//stride +1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)

    img = np.zeros((N, C, H + 2*pad + stride -1, W + 2*pad + stride -1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:,:, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class SimpleCNN:
    def __init__(self, input_dim=(3, 128, 128), conv_param=None, hidden_size=100, output_size=1):
        if conv_param is None:
            conv_param = {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        pool_output_size = filter_num * (conv_output_size // 2) * (conv_output_size // 2)

        self.params = {}
        weight_init_std = np.sqrt(2.0 / (filter_size * filter_size * input_dim[0]))
        self.params['W1'] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params['b1'] = np.zeros(filter_num)

        weight_init_std = np.sqrt(2.0 / pool_output_size)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        weight_init_std = np.sqrt(2.0 / hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=filter_stride, pad=filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SigmoidWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = (y > 0.5).astype(int)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = self.binary_cross_entropy(self.y, self.t)
        return self.loss


    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


    def binary_cross_entropy(self, y, t):
        epsilon = 1e-7
        loss = -np.mean(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon))
        return loss


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # (FN, C, FH, FW)
        self.b = b  # (FN,)
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None


    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        dW = np.dot(self.col.T, dout)
        self.dW = dW.transpose(1,0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h + 2 * self.pad) // self.stride + 1
        out_w = (W - self.pool_w + 2 * self.pad) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max

        return out


    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(-1, pool_size)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx