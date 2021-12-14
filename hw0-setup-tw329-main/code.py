import numpy as np
import matplotlib.pyplot as plt


def sum_numbers(x, y):
    result = x + y
    return result


def multiply_numbers(x, y):
    result = x * y
    return result


def create_add_matrix(x):
    y = np.ones((3,3))
    result = np.add(x, y)
    return result



def indexing_aggregation(x, n):
    result = np.mean(x[:n])
    return result
    """
    TODO: IMPLEMENT ME
    Return the mean value of the first n elements of the input array x.

    Args:
        x (np.ndarray): a 1D numpy array
    Returns:
        output (float): the operation result

    """


def plot_sine(freq, dur, sr, output_path):
    """
    Plot a sine wave with frequency `freq`, duration `dur`, and sample rate `sr`.
    Save to `output`.

    Args:
        freq (float): frequency
        dur (float): duration
        sr (int): sampling rate
        output_path (str): path to save plot

    Returns:
        None

    """
    samples = dur * sr
    y = np.sin(2 * np.pi * freq * np.arange(samples) / sr)
    plt.plot(np.arange(samples) / sr, y)
    plt.xlabel('Time (s)')
    plt.savefig(output_path)
    # no need to implement anything in this function, just make sure you have the required packages installed and
    # generate the output
