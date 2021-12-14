import numpy as np
import pytest

# set random seed for reproducibility
@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)


def test_dft():
    from src.analysis import dft

    for i in range(10):
        x = 2 * np.random.random(512) - 1
        assert np.allclose(np.fft.fft(x), dft(x))

    for i in range(10):
        x = 2 * np.random.random(512) - 1
        x = x + 1j * (2 * np.random.random(512) - 1)
        assert np.allclose(np.fft.fft(x), dft(x))


def test_idft():
    from src.analysis import idft

    for i in range(10):
        x = 2 * np.random.random(512) - 1
        assert np.allclose(x, idft(np.fft.fft(x)))

    for i in range(10):
        X = 2 * np.random.random(512) - 1
        X = X + 1j * (2 * np.random.random(512) - 1)
        assert np.allclose(np.fft.ifft(X), idft(X))
