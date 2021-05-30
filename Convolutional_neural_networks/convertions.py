import numpy as np


def img2col(input_x, filter_h, filter_w, stride, padding):
    N, C, H, W = input_x.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # padding data
    img = np.pad(input_x, [(0, 0), (0, 0), (padding, padding), (padding, padding)])

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y : y_max: stride, x : x_max : stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape((N * out_h * out_w, -1))
    return col


def col2img(input_x, col, filter_h, filter_w, stride, padding):
    N, C, H, W = input_x.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    '''remember to reshape col to a image.
    now col is (N * out_h * out_w, C * filter_h * filter_w)
    => (N, out_h, out_w, C, filter_h, filter_w)
    => (N, C, filter_h, filter_w, out_h, out_w)
    don't directly reshape to the final shape may be encounter some problem
    '''
    col = col.reshape((N, out_h, out_w, C, filter_h, filter_w)).transpose(0, 3, 4, 5, 1, 2)

    '''ceate image tensor:
    we need to accumulate all (out_h, out_w) blocks in col tensor to image tensor
    since we're going to traverse the shape of filter_h and filter_w, so the biggest dimension should be:
    H_img = filter_h - 1 + stride * ((H + 2 * padding - filter_h) / stride + 1) = H + 2 * padding + stride - 1
    W_img = W + w * padding + stride - 1 
    '''
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y : y_max : stride, x : x_max : stride, :, :] += col[:, :, y, x]

    return img 