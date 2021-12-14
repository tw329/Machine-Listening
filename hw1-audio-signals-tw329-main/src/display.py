import numpy as np
import matplotlib.pyplot as plt



def plot_waveform(x, sample_rate):
    """
    TODO: IMPLEMENT ME
    Plot the time domain signal `x`. This is be a time-amplitude plot, in which the x-axis
    is labeled in seconds based on the `sample_rate` argument.

    NOTE: Do not call `plt.show()` within the function.

    Args:
        x (np.array): the 1-dimensional time domain audio signal.
        sample_rate (float): the sample rate in Hz.

    Returns:
    None
    """
    N = x.shape[0]
    t = np.linspace(0, N / sample_rate, N)
    plt.plot(t, x)
    plt.gca()
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()


def plot_spectrum(X, sample_rate):
    """
    TODO: IMPLEMENT ME
    Plot the magnitude spectrum of X. The x-axis should be frequency in Hz. The y-axis is magnitude in decibels.
    Feel free to limit the y-axis (i.e. clip off very low magnitudes)
    Only display frequencies up to and including the Nyquist frequency.

    NOTE: Do not call `plt.show()` within the function.

    Args:
        X (np.array[complex]): A frequency domain signal X, e.g. `X = dft(x)`
        sample_rate (float): sample rate in Hz

    Returns:
    None
    """
    newX = X[:len(X)//2]
    dB = 10 * np.log10(newX)
    frequency = np.linspace(0, sample_rate/2, X.shape[0]/2)
    plt.plot(frequency, dB)
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()
