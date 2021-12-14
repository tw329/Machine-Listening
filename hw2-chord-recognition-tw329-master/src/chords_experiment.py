import numpy as np
import tqdm
import pandas as pd
from .chords import load_data, create_chord_templates, \
    extract_stft_chroma_features, extract_cqt_chroma_features, \
    extract_cens_chroma_features, extract_madmom_chroma_features, \
    compare_chroma_features_to_templates, max_decode, viterbi_decode, \
    evaluate_quantitative, estimate_chords


CONDITIONS = {'stft_max':
                  {'chroma_feature_fn': extract_stft_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': max_decode,
                   'decoding_fn_kwargs': {}},
              'cqt_max':
                  {'chroma_feature_fn': extract_cqt_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': max_decode,
                   'decoding_fn_kwargs': {}},
              'cens_max':
                  {'chroma_feature_fn': extract_cens_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': max_decode,
                   'decoding_fn_kwargs': {}},
              'madmom_max':
                  {'chroma_feature_fn': extract_madmom_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': max_decode,
                   'decoding_fn_kwargs': {}},
              'stft_viterbi':
                  {'chroma_feature_fn': extract_stft_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': viterbi_decode,
                   'decoding_fn_kwargs': {'prob': 0.5}},  # just randomly chosen. may not be best!!
              'cqt_viterbi':
                  {'chroma_feature_fn': extract_cqt_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': viterbi_decode,
                   'decoding_fn_kwargs': {'prob': 0.5}},  # just randomly chosen. may not be best!!
              'cens_viterbi':
                  {'chroma_feature_fn': extract_cens_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': viterbi_decode,
                   'decoding_fn_kwargs': {'prob': 0.5}},  # just randomly chosen. may not be best!!
              'madmom_viterbi':
                  {'chroma_feature_fn': extract_madmom_chroma_features,
                   'chroma_feature_fn_kwargs': {},
                   'decoding_fn': viterbi_decode,
                   'decoding_fn_kwargs': {'prob': 0.5}},  # just randomly chosen. may not be best!!
              }


def run(conditions=CONDITIONS, limit=-1, data_split='validation'):
    """
    Run an experiment in which you extract predictions for a set of conditions.

    Args:
        conditions (dict): Dictionary of conditions that takes the form:
            {<condition_id>:
                   {'chroma_feature_fn': <chroma_feature_fn>,
                    'chroma_feature_fn_kwargs': <chroma_feature_fn_kwargs>,
                    'decoding_fn': <decoding_fn>,
                    'decoding_fn_kwargs': <decoding_fn_kwargs>},
             ... }
        limit (int): Limit the number of tracks to process for quicker debugging. Set to `-1` to process all.
        data_split (str): The data split ('validation' or 'test')

    Returns:
        results (pd.DataFrame): Pandas DataFrame of the results
    """
    results = []
    for condition_id, condition in conditions.items():
        res = run_condition(limit=limit, data_split=data_split, **condition)
        res['condition_id'] = condition_id

        results.append(res)

    results = pd.concat(results)

    return results


def run_condition(chroma_feature_fn, chroma_feature_fn_kwargs, decoding_fn, decoding_fn_kwargs,
                  limit=-1, data_split="validation"):
    """
    Run an experimental condition
    Args:
        chroma_feature_fn: See chords.estimate_chords
        chroma_feature_fn_kwargs: See chords.estimate_chords
        decoding_fn: See chords.estimate_chords
        decoding_fn_kwargs: See chords.estimate_chords
        limit (int): Limit the number of tracks to process for quicker debugging. Set to `-1` to process all.
        data_split (str): The data split ('validation' or 'test')

    Returns:
        results (pd.DataFrame): Pandas DataFrame of the results
    """
    test_tracks, validation_tracks = load_data()

    if data_split == 'validation':
        tracks = validation_tracks
    elif data_split == 'test':
        tracks = test_tracks
    else:
        raise ValueError('Unknown data split.')

    results = []
    for track_id, track in tqdm.tqdm(list(tracks.items())[:limit]):
        est_labels, est_times = estimate_chords(track,
                                                chroma_feature_fn, chroma_feature_fn_kwargs,
                                                decoding_fn, decoding_fn_kwargs)

        res = evaluate_quantitative(track, est_labels, est_times)

        res['track_id'] = track_id

        results.append(res)

    results = pd.DataFrame.from_dict(results)
    return results