import numpy as np
import matplotlib.pyplot as plt
# DO NOT IMPORT ANY OTHER MODULES THAN THOSE SPECIFIED HERE


def dft(x):
    """
    TODO: IMPLEMENT ME
    The Discrete Fourier Transform, which converts a signal of length N from the time domain to the frequency domain.
    The input could be real or complex and the output is complex.

    Implement yourself from the equations in class/book, e.g. DO NOT use np.fft.fft.

    Args:
        x (np.array[float], np.array[complex]): Real or complex input of length N.

    Returns:
        X (np.array[complex]): the complex transformed output of length N
    """
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(M, x)
    return X
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()


def idft(X):
    """
    TODO: IMPLEMENT ME
    The Inverse Discrete Fourier Transform which converts from the frequency domain to the time domain.

    Implement yourself from the equations in class/book, e.g. DO NOT use np.fft.ifft.

    Args:
        X (np.array[complex]): Complex input of length N.

    Returns:
        x (np.array[complex]): the complex transformed output of length N
    """
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    x = 1 / N * np.dot(M, X)
    return x
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()
