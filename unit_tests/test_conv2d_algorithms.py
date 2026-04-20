import numpy as np
from modules.conv2d import Conv2D


def test_conv2d_algorithms_equivalence():
    # Parámetros
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2

    # Entrada controlada
    input_image = np.arange(
        img_height * img_width * in_channels * batch_size,
        dtype=np.float32
    ).reshape(batch_size, in_channels, img_height, img_width)

    # Pesos y bias controlados
    kernels = np.ones(
        (out_channels, in_channels, kernel_size, kernel_size),
        dtype=np.float32
    )
    biases = np.zeros(out_channels, dtype=np.float32)

    # conv_algo = 0 (baseline)
    conv_direct = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=0
    )
    conv_direct.kernels = kernels.copy()
    conv_direct.biases = biases.copy()

    # conv_algo = 1 (im2col numpy)
    conv_im2col = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=1
    )
    conv_im2col.kernels = kernels.copy()
    conv_im2col.biases = biases.copy()

    # conv_algo = 2 (im2col cython)
    conv_im2col_cython = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=2
    )
    conv_im2col_cython.kernels = kernels.copy()
    conv_im2col_cython.biases = biases.copy()

    # Forward
    out_direct = conv_direct.forward(input_image)
    out_im2col = conv_im2col.forward(input_image)
    out_im2col_cython = conv_im2col_cython.forward(input_image)

    # Comprobaciones
    assert np.allclose(out_direct, out_im2col), \
        "Conv2D conv_algo=1 no coincide con conv_algo=0"

    assert np.allclose(out_direct, out_im2col_cython), \
        "Conv2D conv_algo=2 no coincide con conv_algo=0"

    assert np.allclose(out_im2col, out_im2col_cython), \
        "Conv2D conv_algo=1 y conv_algo=2 no coinciden entre sí"

    print("✅ Conv2D algorithms equivalence test passed!")


test_conv2d_algorithms_equivalence()