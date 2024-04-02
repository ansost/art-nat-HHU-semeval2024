"""Compute metrics for train, dev and test monolingual datasets.

NOTE: Make sure that you have run the preprocess_wordlists.py script before running this one.

INPUT must be either the training or dev data for task A (subtaskA_train_monolingual.jsonl or subtaskA_dev_monolingual.jsonl).
Change line 36 according to which dataset you want to use by specifying either 'train' 'dev', or 'test'.

OUTPUT concatenated with the input data is saved as a hdf5 file. 
It is reccommended to load the output file using: 'df = pd.read_hdf("../data/trainA_cheap_metrics.h5", "df")'.

HIDDEN DEPENDENCIES: 
    - tables
    - pyarrow

Usage:
    python computeDataMetrics.py 
"""
import os
import json
import string

import pandas as pd
from tqdm import tqdm
from textstat.textstat import textstat
from nltk.corpus import stopwords


def readability(text) -> float:
    """
    Analyse the readability of the text using textstat readability analysis.

    :param text: the text to be analysed
    :return: the readability of the text
    """
    readability = textstat.flesch_reading_ease(text)
    return readability


def valid_dataset(dataset):
    """Check if dataset is valid."""
    if dataset not in ["dev", "train", "test"]:
        raise ValueError("Dataset must be one of: dev, train, test")


def ensure_preprocessing():
    """Make sure prerequisite data files exist."""
    # Check that directory exists
    assert os.path.exists("../../data/word_lists/")
    files = os.listdir("../../data/word_lists/")
    needed = [
        "enfreq_data.json",
        "fantasy_words.json",
        "wikipedia_words.json",
    ]
    for file in needed:
        if file not in files:
            raise FileNotFoundError(
                f"Required file not found: {file}\n Use the script 'preprocess_wordlists.py to generate the file."
            )


if __name__ == "__main__":
    print("Preparing data...")
    dataset = "test"
    valid_dataset(dataset)
    ensure_preprocessing()

    df = pd.read_json(
        path_or_buf=f"../../data/subtaskA_{dataset}_monolingual.jsonl",
        lines=True,
        engine="pyarrow",
    )
    texts = df["text"].tolist()
    vowels = ["a", "e", "i", "o", "u"]
    stop_words = stopwords.words("english")

    with open("../../data/word_lists/enfreq_data.json", "r") as f:
        top10_content_words, top20_content_words, content_word_freqs = json.load(f)
    with open("../../data/word_lists/fantasy_words.json", "r") as f:
        fantasy_words = json.load(f)
    with open("../../data/word_lists/wikipedia_words.json", "r") as f:
        wikipedia_words = json.load(f)

    all_measures = {}
    (
        TTR_list,
        n_words_list,
        n_punctuation_list,
        avg_word_length_list,
        n_vowels_list,
        readability_list,
        n_content_words_list,
        n_top10_content_words_list,
        n_top20_content_words_list,
        n_fantasy_words_list,
        n_wikipedia_words_list,
        avg_content_word_freq_list,
    ) = ([] for i in range(12))

    print("Computing metrics...")
    for text in tqdm(texts):
        measures = {}
        words = text.split(" ")
        sentences = text.split(".")

        TTR = len(set(words)) / len(words)
        n_words = len(words) / len(sentences)
        n_punctuation = sum([1 for char in text if char in string.punctuation]) / len(
            words
        )
        avg_word_length = sum(len(word) for word in words) / len(words)
        n_vowels = sum([1 for char in text if char in vowels]) / len(words)
        readability_ = readability(text)
        content_words = [word for word in words if word not in stop_words]
        n_content_words = (
            len(content_words) / (len(words) - len(content_words))
            if len(words) != len(content_words)
            else 0
        )
        content_freqs = [
            content_word_freqs[word]
            for word in content_words
            if word in content_word_freqs
        ]
        avg_content_word_freq = (
            sum(float(i) for i in content_freqs) / len(content_freqs)
            if content_freqs
            else 0
        )
        n_top10_content_words = len(top10_content_words) / len(content_words)
        n_top20_content_words = len(top20_content_words) / len(content_words)
        n_fantasy_words = len(fantasy_words) / len(content_words)
        n_wikipedia_words = len(wikipedia_words) / len(content_words)

        TTR_list.append(TTR)
        n_words_list.append(n_words)
        n_punctuation_list.append(n_punctuation)
        avg_word_length_list.append(avg_word_length)
        n_vowels_list.append(n_vowels)
        readability_list.append(readability_)
        n_content_words_list.append(n_content_words)
        n_top10_content_words_list.append(n_top10_content_words)
        n_top20_content_words_list.append(n_top20_content_words)
        n_fantasy_words_list.append(n_fantasy_words)
        n_wikipedia_words_list.append(n_wikipedia_words)
        avg_content_word_freq_list.append(avg_content_word_freq)

    print("Saving metrics...")
    measures = {
        "TTR": TTR_list,
        "n_words": n_words_list,
        "n_punctuation": n_punctuation_list,
        "avg_word_length": avg_word_length_list,
        "n_vowels": n_vowels_list,
        "readability": readability_list,
        "ratio_content_words": n_content_words_list,
        "ratio_top10_content_words": n_top10_content_words_list,
        "ratio_top20_content_words": n_top20_content_words_list,
        "ratio_fantasy_words": n_fantasy_words_list,
        "ratio_wikipedia_words": n_wikipedia_words_list,
    }

    df_measures = pd.DataFrame.from_dict(measures)
    df_measures = pd.concat([df, df_measures], axis=1)
    df_measures.to_hdf(
        f"../../data/feature_metrics/{dataset}A_cheap_metrics.h5", key="df", mode="w"
    )
    print(df_measures[df_measures.columns[-5:]].head())
