import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import accuracy_score
from .instrument_recognizer import load_data, extract_mfcc_features, compute_summary_statistics


def run(conditions, test_split='validation'):
    """
    Run an experiment in which you extract predictions for a set of conditions.

    Args:
        conditions (dict): Dictionary of conditions that takes the form:
            {<condition_id>:
                   {'feature_fn': <feature_fn>,
                    'feature_fn_kwargs': <feature_fn_kwargs>,
                    'summary_fn': <summary_fn>,
                    'summary_fn_kwargs': <summary_fn_kwargs>,
                    'preprocessor': <preprocessor>
                    'classifier_fn': <classifier_fn>,
                    'classifier_kwargs': <classifier_kwargs>,
                    'load_data_kwargs': <load_data_kwargs>}
             ... }
        test_split (str): The test data split ('validation' or 'test')

    Returns:
        results (pd.DataFrame): Pandas DataFrame of the results
    """
    results = []
    for condition_id, condition in tqdm.tqdm(conditions.items()):
        res = run_condition(test_split=test_split, **condition)
        res['condition_id'] = condition_id

        results.append(res)

    results = pd.DataFrame.from_dict(results)

    return results


def run_condition(feature_fn, feature_fn_kwargs,
                  summary_fn, summary_fn_kwargs,
                  preprocessor,
                  classifier_fn, classifier_kwargs,
                  test_split="validation",
                  return_predictions=False,
                  load_data_kwargs=None):
    """
    Run an experimental condition
    Args:
        feature_fn: Fn to extract features
        feature_fn_kwargs: Args for feature_fn
        summary_fn: Fn for summarizing features for a clip
        summary_fn_kwargs: Args for summary_fn
        classifier_fn: See Fn for classifying
        classifier_kwargs: Args for classifier_fn
        preprocessor: See sklearn.pipeline.Pipeline
        test_split (str): The test data split ('validation' or 'test')
        return_predictions (bool): If True, return `results`, `test_labels`, `pred_labels`
        load_data_kwargs: Args for load_data

    Returns:
        results (pd.DataFrame): Pandas DataFrame of the results
    """
    if load_data_kwargs is None:
        load_data_kwargs = dict()
    training_tracks, test_tracks, validation_tracks = load_data(**load_data_kwargs)

    if test_split == 'validation':
        test_tracks = validation_tracks
    elif test_split == 'test':
        pass
    else:
        raise ValueError('Unknown data split.')

    training_labels = np.array([t.instrument for t in training_tracks.values()])
    test_labels = np.array([t.instrument for t in test_tracks.values()])

    training_feats = feature_fn(training_tracks, **feature_fn_kwargs)
    training_feats = summary_fn(training_feats, **summary_fn_kwargs)

    test_feats = feature_fn(test_tracks, **feature_fn_kwargs)
    test_feats = summary_fn(test_feats, **summary_fn_kwargs)

    classifier = classifier_fn(**classifier_kwargs)
    preprocessor.fit(training_feats, training_labels)
    classifier.fit(preprocessor.transform(training_feats), training_labels)
    pred_labels = classifier.predict(preprocessor.transform(test_feats))
    result = dict(accuracy_score=accuracy_score(test_labels, pred_labels))

    if return_predictions:
        return result, test_labels, pred_labels
    else:
        return result
