import random

import numpy as np
import librosa
import mirdata

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def sample_n_instrument_examples(subset, n_examples):
    instruments = set([v.instrument for v in subset.values()])

    sample = []
    for instrument in instruments:
        instrument_tracks = [(k, v) for k, v in subset.items() if v.instrument == instrument]
        if n_examples != -1:
            sample.extend(random.sample(instrument_tracks, n_examples))
        else:
            sample.extend(instrument_tracks)

    return dict(sample)


def load_data(training_limit_per_instrument=120,
              test_limit_per_instrument=120,
              validation_limit_per_instrument=25,
              seed=485689):
    """
    Load the Medley-solos-DB data using `mirdata`. Evenly sample the number of examples for each instrument.

    See https://zenodo.org/record/1344103#.YFjbn2RueLo for more details about the dataset.

    Args:
        training_limit_per_instrument (int)
        test_limit_per_instrument (int)
        validation_limit_per_instrument (int)
        seed (int)

    Returns:
        training_tracks (dict[mirdata.guitarset.Track])
        test_tracks (dict[mirdata.guitarset.Track])
        validation_tracks (dict[mirdata.guitarset.Track])
    """
    # fix the seed for reproducibility
    random.seed(seed)

    medley_solos_db = mirdata.initialize('medley_solos_db')
    # medley_solos_db.download(force_overwrite=True, cleanup=True)
    tracks = medley_solos_db.load_tracks()

    training_tracks = dict([(k, v) for k, v in tracks.items() if v.subset == 'training'])
    test_tracks = dict([(k, v) for k, v in tracks.items() if v.subset == 'test'])
    validation_tracks = dict([(k, v) for k, v in tracks.items() if v.subset == 'validation'])

    training_tracks = sample_n_instrument_examples(training_tracks, training_limit_per_instrument)
    test_tracks = sample_n_instrument_examples(test_tracks, test_limit_per_instrument)
    validation_tracks = sample_n_instrument_examples(validation_tracks, validation_limit_per_instrument)

    return training_tracks, test_tracks, validation_tracks


def extract_mfcc_features(tracks, **kwargs):
    """
    Extract MFCCs for a set of tracks from mirdata.

    Args:
        tracks (dict[mirdata.guitarset.Track])
        **kwargs: keyword arguments to pass to `extract_mfcc_features_for_track`.

    Returns:
        mfcc_features (np.ndarray [shape=(n_tracks, n_mfccs, n_frames)])
    """
    output = []

    for k, t in tracks.items():
        output.append(extract_mfcc_features_for_track(t.audio_path, **kwargs))

    return np.array(output)


def extract_mfcc_features_for_track(audio_file_path, n_mfcc, hop_length=512, sr=44100,
                                    include_dc=True, include_deltas=False, include_delta_deltas=False,
                                    **kwargs):
    """
    Open the audio at sample rate `sr` and extract mfccs using `librosa.feature.mfcc`.
    Calculate 1st and 2nd derivative approximations using `librosa.feature.delta`.

    Args:
        audio_file_path (str): Path to audio file.
        n_mfcc (int): Number of mfccs to extract. Note this is not the same as the number of number of mel bands.
        hop_length (int)
        include_dc (bool): If False, remove the first MFCC coefficient.
        include_deltas (bool): If True, append the deltas (approximation of 1st derivative) to the MFCCs
        include_delta_deltas (bool): If True, append the delta-deltas (approximation of 2nd derivative) to the MFCCs
        **kwargs (dict): Additional keyword arguments that should be passed to `librosa.feature.mfcc`.

    Returns:
        mfcc_features (np.ndarray [shape=(n_features, n_frames)])
    """
    x, sr = librosa.load(audio_file_path, sr=sr)

    mfcc_features = librosa.feature.mfcc(x, n_mfcc=n_mfcc, hop_length=hop_length, sr=sr, **kwargs)
    if not include_dc:
        output = mfcc_features[1:, :]
    else:
        output = mfcc_features

    if include_deltas:
        mfcc_features_deltas = librosa.feature.delta(mfcc_features, order=1)
        output = np.vstack([output, mfcc_features_deltas])

    if include_delta_deltas:
        mfcc_features_delta_deltas = librosa.feature.delta(mfcc_features, order=2)
        output = np.vstack([output, mfcc_features_delta_deltas])

    return output


def compute_summary_statistics(features,
                               include_mean=False, include_variance=False,
                               include_max=False, include_min=False):
    """
    Compute summary statistics over the time axis.

    Args:
        features (np.ndarray([n_tracks, n_features, n_frames)]): Input features to summarize.
        include_mean (bool):
        include_variance (bool):
        include_max (bool):
        include_min (bool):

    Returns:
        summarized_features (np.ndarray([n_tracks, n_features * n_stats)])
    """
    if not include_mean and not include_variance and not include_max and not include_min:
        raise ValueError('At least one "include" argument must be True.')

    output = []

    if include_mean:
        output.append(np.mean(features, axis=-1))

    if include_variance:
        output.append(np.var(features, axis=-1))

    if include_max:
        output.append(np.max(features, axis=-1))

    if include_min:
        output.append(np.min(features, axis=-1))

    return np.hstack(output)


def invert_mfccs(mfccs, n_mels=128, hop_length=512, sr=44100):
    """
    Wrapper for using `librosa.feature.inverse.mfcc_to_audio` invert mfccs.
    """
    return librosa.feature.inverse.mfcc_to_audio(mfccs, n_mels=n_mels, hop_length=hop_length, sr=sr, dct_type=2,
                                                 norm='ortho', ref=1.0, lifter=0)
