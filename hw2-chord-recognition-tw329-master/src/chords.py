import numpy as np
import librosa
import mirdata
import mir_eval
import tqdm
import pathlib
import os
import re
from . import package_directory
from scipy.special import softmax
from scipy.spatial.distance import cdist


def load_data():
    """
    Load the GuitarSet data using `mirdata`. Only the audio_mic data is downloaded and the returned
    data is limited to the "accompaniment" (i.e. "comp") recordings.

    The data is then split by chord progression into test (2/3) and validation sets (1/3).

    See https://guitarset.weebly.com/ for more details about the dataset.

    Returns:
        test_tracks (dict[mirdata.guitarset.Track])
        validation_tracks (dict[mirdata.guitarset.Track])
    """
    guitarset = mirdata.initialize('guitarset')
    guitarset.download(partial_download=['audio_mic', 'annotations'])
    tracks = guitarset.load_tracks()
    tracks = dict([(k, v) for k, v in tracks.items() if 'comp' in k])

    # split by chord progression
    pat = r"\d+_[a-zA-z]+(\d).+"
    prog = re.compile(pat)
    test_tracks = dict([(k, v) for k, v in tracks.items() if prog.match(k).group(1) in ['1', '2']])
    validation_tracks = dict([(k, v) for k, v in tracks.items() if prog.match(k).group(1) == '3'])

    return test_tracks, validation_tracks


def create_chord_templates():
    """
    TODO: IMPLEMENT ME

    Create the chord templates for all major triads, minor triads, and the "no-chord" templates (all ones).

    Each template should consist of a numpy array of length 12 (one dimension for each pitch class, starting
    on C) with a `1` if the pitch class is present and a `0` otherwise.

    The chord labels should be in the form "<Root>:<maj|min>". The "no-chord" label should be "N".
    Note that for chords that have enharmonic equivalents, e.g. "C#:maj" and "Db:maj", only include
    the "sharp" variant, i.e. "C#:maj" from that example.

    For example, the template for "C:maj" should be [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    All templates should be concatenated together to form a matrix of size (12,25).

    Returns:
        chord_labels (np.ndarray [dtype=str, shape=(25,)]): The labels for each chord, e.g. "C:maj" or "A#:min". Should be length 25.
        chord_templates (np.ndarray [shape=(12, 25)]): The chord templates, concatenated together into a matrix.
    """
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    maj_min = ["maj", "min"]
    #chord labels
    chord_labels = []
    for i in range(len(maj_min)):
        for j in range(len(chords)):
            chord_labels.append(chords[j]+":"+maj_min[i])
    chord_labels.append("N")
    chord_labels = np.array(chord_labels)
    #chord templates
    N_templates = [1] * len(chords)
    t_templates = [0] * len(chords)
    chord_templates = []
    for i in range(len(chord_labels)):
        if i < 12:
            t_templates[(0 + i) % (len(chords))] = 1
            t_templates[(4 + i) % (len(chords))] = 1
            t_templates[(7 + i) % (len(chords))] = 1
            chord_templates.append(t_templates)
            t_templates = [0] * len(chords)
        elif i < len(chord_labels)-1:
            t_templates[(0 + i) % (len(chords))] = 1
            t_templates[(3 + i) % (len(chords))] = 1
            t_templates[(7 + i) % (len(chords))] = 1
            chord_templates.append(t_templates)
            t_templates = [0] * len(chords)
        else:
            chord_templates.append(N_templates)            
    chord_templates = np.transpose(np.array(chord_templates))
    # replace the following line with an actual implementation that returns something
    return chord_labels, chord_templates


def extract_cens_chroma_features(audio_file_path, hop_length=512, sr=22050, **kwargs):
    """
    TODO: IMPLEMENT ME

    Open the audio at sample rate `sr` and extract chroma using `librosa.feature.chroma_cens`.
    Also, calculate the times array (when the beginning of each chroma feature frame occurs).

    Args:
        audio_file_path (str): Path to audio file.
        hop_length (int):
        sr (int): Sample rate
        **kwargs (dict): Additional keyword arguments that should be passed to `librosa.feature.chroma_cens`.

    Returns:
        times (np.ndarray): The array of times of each frame in seconds.
        chroma_features (np.ndarray [shape=(n_states, n_frames)] ):
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, **kwargs)
    times = librosa.times_like(chroma_cens)
    # replace the following line with an actual implementation that returns something
    return times, chroma_cens


def extract_stft_chroma_features(audio_file_path, hop_length=512, sr=22050, **kwargs):
    """
    TODO: IMPLEMENT ME

    Open the audio at sample rate `sr` and extract chroma using `librosa.feature.chroma_stft`.
    Also, calculate the times array (when the beginning of each chroma feature frame occurs).

    Args:
        audio_file_path (str): Path to audio file.
        hop_length (int):
        sr (int): Sample rate
        **kwargs (dict): Additional keyword arguments that should be passed to `librosa.feature.chroma_stft`.

    Returns:
        times (np.ndarray): The array of times of each frame in seconds.
        chroma_features (np.ndarray [shape=(n_states, n_frames)] ):
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, **kwargs)
    times = librosa.times_like(chroma_stft)
    # replace the following line with an actual implementation that returns something
    return times, chroma_stft


def extract_cqt_chroma_features(audio_file_path, hop_length=512, sr=22050, **kwargs):
    """
    TODO: IMPLEMENT ME

    Open the audio at sample rate `sr` and extract chroma using `librosa.feature.chroma_cqt`.
    Also, calculate the times array (when the beginning of each chroma feature frame occurs).

    Args:
        audio_file_path (str): Path to audio file.
        hop_length (int):
        sr (int): Sample rate
        **kwargs (dict): Additional keyword arguments that should be passed to `librosa.feature.chroma_cqt`.

    Returns:
        times (np.ndarray): The array of times of each frame in seconds.
        chroma_features (np.ndarray [shape=(n_states, n_frames)] ):
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, **kwargs)
    times = librosa.times_like(chroma_cqt)
    # replace the following line with an actual implementation that returns something
    return times, chroma_cqt


def extract_madmom_chroma_features(audio_file_path, hop_length=None, sr=None, **kwargs):
    """
    PRECOMPUTED SINCE SOME PEOPLE HAD TROUBLE WITH MADMOM.
    Note that madmom uses a different hop_length than librosa, so that times array is different.

    Read in pre-computed features

    Returns:
        times (np.ndarray): The array of times of each frame in seconds.
        chroma_features (np.ndarray [shape=(n_states, n_frames)] )
    """
    arrays = np.load(os.path.join(package_directory, 'madmom_precompute', os.path.basename(audio_file_path) + '.npz'))
    times = arrays['times']
    chroma_features = arrays['chroma_features']

    return times, chroma_features


def compare_chroma_features_to_templates(chroma_features, templates):
    """
    TODO: IMPLEMENT ME

    Compare each frame in `chroma_features` to each template in `templates` using cosine similarity.
    Normalize the similarities computed in each frame to probabilities using the softmax function
    (see https://en.wikipedia.org/wiki/Softmax_function)

    Note that a chroma_feature that is a zero vector will result in NaNs due to the denominator in
    the cosine similarity. Make sure you account for this in a sensible way so that no NaNs occur.

    Args:
        chroma_features (np.ndarray [shape=(n_chroma, n_frames)]): Input chroma features
        templates (np.ndarray [shape=(12,25)]: Chord templates from create_chord_templates()

    Returns:
        chord_likelihoods (np.ndarray [shape=(25, n_frames)]): The chord likelihoods given the chroma observations
    """
    chord_likelihoods = []
    for i in range(len(templates[0])):
        y_list = []
        for j in range(len(chroma_features[0])):
            C = chroma_features[:,j]
            T = templates[:, i]
            y_list.append(np.dot(C, T)/(np.linalg.norm(C)*np.linalg.norm(T)))
        chord_likelihoods.append(y_list)
    chord_likelihoods = np.array(chord_likelihoods)
    chord_likelihoods = softmax(chord_likelihoods)
    # replace the following line with an actual implementation that returns something
    return chord_likelihoods


def max_decode(chord_likelihoods, chord_labels, **kwargs):
    """
    TODO: IMPLEMENT ME

    'Decode' the `chord_likelihoods` by returning the chord_label with the highest likelihood
    in each time frame.

    The output should be a list of chord labels, e.g. ["C:maj", "A:min", "A#:min"]

    Args:
        chord_likelihoods (np.ndarray [shape=(25, n_frames)]): Matrix of the chord likelihoods estimated
            for each time frame.
        chord_labels (np.ndarray [shape=(25,)]: Set of chord labels from `create_chord_templates()`
        **kwargs: Unused for this function

    Returns:
    est_labels (list[str]): List of the predicted chord labels for each frame. Length should be number the
        same number of frames as the `chord_likelihoods`.
    """
    est_labels = []
    for i in range(len(chord_likelihoods[0])):
        highest_index = np.argmax(chord_likelihoods[:, i])
        est_labels.append(chord_labels[highest_index])
    # replace the following line with an actual implementation that returns something
    return est_labels


def viterbi_decode(chord_likelihoods, chord_labels, prob, **kwargs):
    """
    TODO: IMPLEMENT ME

    'Decode' the `chord_likelihoods` by returning the chord_label using viterbi decoding.
    Use the functions in the `librosa.sequence` module to help with this.

    Construct a transition matrix that has a self-transition probability of `prob` along the
    diagonal uniform transtion probability on the off-diagonal.

    The output should be a list of chord labels, e.g. ["C:maj", "A:min", "A#:min"]

    Args:
        chord_likelihoods (np.ndarray [shape=(25, n_frames)]): Matrix of the chord likelihoods estimated
            for each time frame.
        chord_labels (np.ndarray [shape=(25,)]: Set of chord labels from `create_chord_templates()`
        prob (float): Probability of self-transition.
        **kwargs (dict): Additional keyword arguments to pass to the viterbi function.

    Returns:
    est_labels (list[str]): List of the predicted chord labels for each frame. Length should be number the
        same number of frames as the `chord_likelihoods`.
    """
    transition = librosa.sequence.transition_loop(25, prob)
    path = librosa.sequence.viterbi(chord_likelihoods, transition)
    est_labels = []
    for i in range(len(path)):
        est_labels.append(chord_labels[path[i]])
    # replace the following line with an actual implementation that returns something
    return est_labels


def estimate_chords(track, chroma_feature_fn, chroma_feature_fn_kwargs,
                    decoding_fn, decoding_fn_kwargs):
    """
    TODO: IMPLEMENT ME

    Estimate chords for a given track.

    Put it all together:
        1. Create the chord templates
        2. Extract chroma features with `chroma_feature_fn`, passing in any extra keywords with `chroma_feature_fn_kwargs`
        3. Compare chroma to templates
        4. Decode likelihood sequence with `decoding_fn`, passing in any extra keywords with `decoding_fn_kwargs`

    Args:
        track (mirdata.guitarset.Track): The track to estimate chords for
        chroma_feature_fn (function): The function to use to extract chroma features.
        chroma_feature_fn_kwargs (dict): Any extra keywords to pass to `chroma_feature_fn`.
        decoding_fn (function): The function to use for decoding the likelihood sequence.
        decoding_fn_kwargs (dict): Any extra keywords to pass to `decoding_fn`.

    Returns:
        est_labels (list[str]): List of the predicted chord labels for each frame.
        est_times (np.ndarray): The array of times of each frame in seconds.
    """
    chord_labels, chord_templates = create_chord_templates()
    est_times, chroma_features = chroma_feature_fn(track.audio_mic_path, **chroma_feature_fn_kwargs)
    likelihoods = compare_chroma_features_to_templates(chroma_features, chord_templates)
    est_labels = decoding_fn(likelihoods, chord_labels, **decoding_fn_kwargs)
    # replace the following line with an actual implementation that returns something
    return est_labels, est_times


def evaluate_quantitative(track, est_labels, est_times):
    """
    Evaluate the output of the chord recognition models using `mir_eval`.
    This will output a dictionary of the accuracy of roots, major/minor, etc.
    See https://craffel.github.io/mir_eval/ for details.

    Args:
        track (mirdata.guitarset.Track): The GuitarSet track to use as reference.
        est_labels (list[str]): List of the predicted chord labels for each frame.
        est_times (np.ndarray [shape=(n_frames,)]): The time of each frame.

    Returns:
        results (dict): Dictionary containing accuracy values.
    """
    ref_intervals = track.inferred_chords.intervals
    ref_labels = track.inferred_chords.labels

    est_intervals = mir_eval.util.boundaries_to_intervals(np.append(est_times,
                                                                    ref_intervals.max()))

    results = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
    return results


def evaluate_qualitative(track, est_labels, est_times, sr=44100):
    """
    TODO: IMPLEMENT ME

    Qualitative evaluation through sonification. Use the `mir_eval.sonify.chords` function
    to sonify the estimated chords. Also load the audio_mic audio. Note that these
    should be synthesized and read in at the same sample rate, `sr`.

    Return the sonification as the first element of a list, and the track audio in the second element of
    a list such that when passed to `IPython.display.Audio`, the sonification will be in the left channel,
    and the track audio will be in the right channel.

    Args:
        track (mirdata.guitarset.Track): The GuitarSet track to use as reference.
        est_labels (list[str]): List of the predicted chord labels for each frame.
        est_times (np.ndarray [shape=(n_frames,)]): The time of each frame.
        sr (int): sample rate to read and synthesize audio

    Returns:
        left_channel (np.ndarray): Audio of left channel
        right_channel (np.ndarray): Audio of right channel
    """
    right, sample_rate= librosa.load(track.audio_mic_path, sr=sr)
    interval = []
    for i in range(len(est_times)-1):
        interval.append([est_times[i], est_times[i+1]])
    interval.append([est_times[-1], est_times[-1]+(est_times[-1]-est_times[-2])])
    left = mir_eval.sonify.chords(est_labels, np.array(interval), sr, length=len(right))
    left_channel = left
    right_channel = right
    # replace the following line with an actual implementation that returns something
    return left_channel, right_channel

