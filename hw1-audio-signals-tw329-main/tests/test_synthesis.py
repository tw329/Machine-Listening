import numpy as np
import pytest

# set random seed for reproducibility
@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)


def test_midi_to_frequency():
    from src.synthesis import midi_to_frequency

    assert midi_to_frequency(60, 440.) == pytest.approx(261.63, 0.01)
    assert midi_to_frequency(22, 440.) == pytest.approx(29.135, 0.01)
    assert midi_to_frequency(69, 450.) == pytest.approx(450.)
    assert midi_to_frequency(81, 450.) == pytest.approx(900.)
    assert midi_to_frequency(57, 430.) == pytest.approx(215.)


def test_generate_envelope():
    from src.synthesis import generate_envelope

    for i in range(1000):
        sample_rate = np.random.choice([22050, 44100, 48000])
        attack_time = np.random.random() + 0.05
        decay_time = np.random.random() + 0.05
        envelope = generate_envelope(attack_time, decay_time, sample_rate)
        assert envelope[0] == 0
        assert np.max(envelope) == 1
        assert np.sum(envelope == 1) == 1
        assert envelope.shape[0] == int(round((attack_time + decay_time) * sample_rate))
        assert np.min(envelope) == 0
        assert envelope[int(0.5 * attack_time * sample_rate)] < np.max(envelope)
        assert envelope[int(0.5 * attack_time * sample_rate)] > np.min(envelope)
        assert envelope[int((attack_time + 0.5 * decay_time) * sample_rate)] < np.max(envelope)
        assert envelope[int((attack_time + 0.5 * decay_time) * sample_rate)] > np.min(envelope)


def test_generate_complex_tone():
    from src.synthesis import generate_complex_tone

    for i in range(100):
        sample_rate = np.random.choice([22050, 44100, 48000])
        freq = np.random.random() * 1000
        assert generate_complex_tone(freq, [[1, 1, 0],], 1, sample_rate).shape[0] == sample_rate
        assert generate_complex_tone(freq, [[1, 1, 0], [2, 1, 0]], 1, sample_rate).shape[0] == sample_rate
        assert generate_complex_tone(2 * freq, [[1, 1, 0], ], 2, sample_rate).shape[0] == (2 * sample_rate)
        assert np.allclose(generate_complex_tone(freq, [[1, 1, 0],], 1, sample_rate),
                           np.sin(2*np.pi*freq*np.arange(sample_rate)/sample_rate))
        assert np.allclose(generate_complex_tone(freq, [[1, 1, 0], [1, 1, 0]], 1, sample_rate),
                           2 * np.sin(2*np.pi*freq*np.arange(sample_rate)/sample_rate))
        assert np.allclose(generate_complex_tone(freq, [[1, 1, 0], [1, 1, np.pi]], 1, sample_rate),
                           np.zeros(sample_rate))
