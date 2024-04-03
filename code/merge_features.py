"""Merge all copmuted features together in one file and scale all numerical columns.
This script requires slight renaming of the files uploaded by Wiebke to facilitate loading and dataset selection.

CHOOSE your dataset by specifying it in line 34.

Usage:
    merge_features.py
"""

import pandas as pd


def validate_dataset(dataset):
    """Check if dataset is valid."""
    if dataset not in ["dev", "train", "test"]:
        raise ValueError("Dataset must be one of: dev, train, test")


def remove_cols(df):
    """Remove columns that are not needed."""
    cols = ["text", "label", "model", "source", "id"]

    for col in cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    # Choose dataset from: dev, train, test
    dataset = "test"
    validate_dataset(dataset)

    cheap_features = pd.read_hdf(
        f"../../data/feature_metrics/{dataset}A_cheap_metrics.h5", "df"
    )
    cheap_features = remove_cols(cheap_features)

    wiebke_features = pd.read_pickle(
        f"../../data/feature_metrics/{dataset}A_wiebke_metrics.pkl"
    )
    wiebke_features = remove_cols(wiebke_features)

    expensive_features = pd.read_hdf(
        f"../../data/feature_metrics/{dataset}A_expensive_metrics.h5", "df"
    )
    expensive_features = remove_cols(expensive_features)

    df = pd.concat(
        [expensive_features, cheap_features, wiebke_features],
        axis=1,
    )

    df = df[
        [
            "max_depth",
            "mean_depth",
            "verb_noun_ratio",
            "n_negation_words",
            "avg_passive_constructions",
            "TTR",
            "n_words",
            "n_punctuation",
            "n_unique_words",
            "avg_word_length",
            "n_vowels",
            "readability",
            "ratio_content_words",
            "ratio_top10_content_words",
            "ratio_top20_content_words",
            "ratio_fantasy_words",
            "ratio_wikipedia_words",
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
            "NEGATIVE",
            "POSITIVE",
            "informal",
            "formal",
            "toxic",
            "non_toxic",
        ]
    ]
    df.reset_index(inplace=True)
    df.to_hdf(
        f"../../data/feature_metrics/{dataset}A_all_metrics.h5",
        key="df",
        mode="w",
        index=False,
    )
