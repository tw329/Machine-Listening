import numpy as np
import pytest
import os
import mir_eval
import soundfile as psf

TEST_FILE = 'data/test_file.wav'

# set random seed for reproducibility
@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)


def test_create_chord_templates():
    from src.chords import create_chord_templates

    chord_labels, chord_templates = create_chord_templates()

    # check properties
    assert np.allclose(chord_templates.sum(axis=1), np.ones(12) * 7)
    assert np.allclose(chord_templates[:, np.array(chord_labels) != "N"].sum(axis=0), np.ones(24) * 3)

    # check an individual chord
    assert chord_templates[:, chord_labels.tolist().index("C:maj")][4] == 1

    # make sure no flats in chord_labels
    assert np.all(['b' not in c for c in chord_labels])


def test_extract_stft_chroma_features():
    from src.chords import extract_stft_chroma_features

    times, chroma_features = extract_stft_chroma_features(TEST_FILE)
    assert np.array_equal(chroma_features.argmax(axis=0), np.ones(chroma_features.shape[1]) * 9)

    assert times[0] == 0
    assert 0.99 == pytest.approx(times[-1], rel=1e-2)


def test_extract_cqt_chroma_features():
    from src.chords import extract_cqt_chroma_features

    times, chroma_features = extract_cqt_chroma_features(TEST_FILE)
    assert np.array_equal(chroma_features.argmax(axis=0), np.ones(chroma_features.shape[1]) * 9)

    assert times[0] == 0
    assert 0.99 == pytest.approx(times[-1], rel=1e-2)


def test_extract_cens_chroma_features():
    from src.chords import extract_cens_chroma_features

    times, chroma_features = extract_cens_chroma_features(TEST_FILE)

    assert np.array_equal(chroma_features.argmax(axis=0), np.ones(chroma_features.shape[1]) * 9)

    assert times[0] == 0
    assert 0.99 == pytest.approx(times[-1], rel=1e-2)


def test_compare_chroma_features_to_templates():
    from src.chords import compare_chroma_features_to_templates, create_chord_templates

    chord_labels, chord_templates = create_chord_templates()

    chroma_features = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]).T

    likelihoods = compare_chroma_features_to_templates(chroma_features, chord_templates)

    assert likelihoods.shape[1] == 1
    assert np.argmax(likelihoods[:, 0]) == list(chord_labels).index("C:maj")
    assert 1 == pytest.approx(np.sum(likelihoods[:, 0]))
    assert np.allclose(compare_chroma_features_to_templates(chroma_features, chord_templates),
                       compare_chroma_features_to_templates(2 * chroma_features, chord_templates),)


def test_max_decode():
    from src.chords import max_decode, create_chord_templates

    chord_labels, chord_templates = create_chord_templates()

    chord_likelihoods = np.ones([25, 4]) * 0.1
    for i in np.arange(4):
        chord_likelihoods[i, i] = 1
    chord_likelihoods = chord_likelihoods / chord_likelihoods.sum(axis=0, keepdims=True)

    assert np.array_equal(max_decode(chord_likelihoods, chord_labels), [chord_labels[0],
                                                                        chord_labels[1],
                                                                        chord_labels[2],
                                                                        chord_labels[3]])

    chord_likelihoods = np.ones([25, 4]) * 0.1
    for i in np.arange(4):
        chord_likelihoods[0, i] = 1
    chord_likelihoods = chord_likelihoods / chord_likelihoods.sum(axis=0, keepdims=True)

    assert np.array_equal(max_decode(chord_likelihoods, chord_labels), [chord_labels[0],
                                                                        chord_labels[0],
                                                                        chord_labels[0],
                                                                        chord_labels[0]])

    chord_likelihoods = np.ones([25, 4]) * 0.1
    chord_likelihoods[0, 0] = 0.9
    chord_likelihoods[1, 0] = 0.8
    chord_likelihoods[0, 1] = 0.8
    chord_likelihoods[1, 1] = 0.9
    chord_likelihoods[0, 2] = 0.9
    chord_likelihoods[1, 2] = 0.8
    chord_likelihoods[0, 3] = 0.8
    chord_likelihoods[1, 3] = 0.9
    chord_likelihoods = chord_likelihoods / chord_likelihoods.sum(axis=0, keepdims=True)

    assert np.array_equal(max_decode(chord_likelihoods, chord_labels), [chord_labels[0],
                                                                        chord_labels[1],
                                                                        chord_labels[0],
                                                                        chord_labels[1]])


def test_viterbi_decode():
    from src.chords import viterbi_decode, create_chord_templates

    chord_labels, chord_templates = create_chord_templates()

    chord_likelihoods = np.ones([25, 4]) * 0.1
    chord_likelihoods[0, 0] = 0.9
    chord_likelihoods[1, 0] = 0.8
    chord_likelihoods[0, 1] = 0.8
    chord_likelihoods[1, 1] = 0.9
    chord_likelihoods[0, 2] = 0.9
    chord_likelihoods[1, 2] = 0.8
    chord_likelihoods[0, 3] = 0.8
    chord_likelihoods[1, 3] = 0.9
    chord_likelihoods = chord_likelihoods / chord_likelihoods.sum(axis=0, keepdims=True)

    assert np.array_equal(viterbi_decode(chord_likelihoods, chord_labels, 0.01), [chord_labels[0],
                                                                                  chord_labels[1],
                                                                                  chord_labels[0],
                                                                                  chord_labels[1]])

    assert np.array_equal(viterbi_decode(chord_likelihoods, chord_labels, 0.9), [chord_labels[0],
                                                                                 chord_labels[0],
                                                                                 chord_labels[0],
                                                                                 chord_labels[0]])


def test_estimate_chords():
    from src.chords import estimate_chords, create_chord_templates, extract_cens_chroma_features, max_decode

    class Track:
        def __init__(self, audio_mic_path):
            self.audio_mic_path = audio_mic_path

    sr = 44100
    path = 'data/_temp.wav'
    chroma_feature_fn = extract_cens_chroma_features
    chroma_feature_fn_kwargs = {}
    decoding_fn = max_decode
    decoding_fn_kwargs = {}

    chord_labels, _ = create_chord_templates()
    chord_labels = [c for c in chord_labels if c != "N"]

    for c in chord_labels:
        x = mir_eval.sonify.chords([c], np.array([[0, 2]]), sr)
        psf.write(path, x, sr)
        track = Track(path)
        est_labels, est_times = estimate_chords(track, chroma_feature_fn, chroma_feature_fn_kwargs,
                                                decoding_fn, decoding_fn_kwargs)
        os.remove(path)

        assert np.all(np.array(est_labels) == c)
        assert np.all(est_times <= 2)


def test_evaluate_qualitative():
    from src.chords import evaluate_qualitative, load_data

    track_id, track = list(load_data()[0].items())[0]

    est_times, est_labels = mir_eval.util.intervals_to_samples(track.inferred_chords.intervals,
                                                               track.inferred_chords.labels,)
    left, right = evaluate_qualitative(track, est_labels, est_times, sr=44100)

    assert left.ndim == 1
    assert right.ndim == 1
    assert left.shape[0] == right.shape[0]

