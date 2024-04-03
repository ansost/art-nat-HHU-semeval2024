"""Compute language level (CEFR) and POS features for a given text.
Select the dataset to use in line 18. 

Usage:
    python cefr_pos_features.py
"""

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, pos_tag_sents, sent_tokenize
from nltk.stem.porter import *

nltk.download("wordnet")

if __name__ == "__main__":

    dataset = "train"
    df = pd.read_json(
        f"../../data/subtaskA_{dataset}_monolingual.jsonl", lines=True, engine="pyarrow"
    )
    cefrj = pd.read_csv("../../data/cefrj-vocabulary-profile-1.5.csv")
    cefrj_c = pd.read_csv("../../data/octanove-vocabulary-profile-c1c2-1.0.csv")
    cefrj = pd.concat([cefrj, cefrj_c], axis=0)

    # Preprocess text
    nltk.download("averaged_perceptron_tagger")
    nltk.download("universal_tagset")
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag_sents([tokens], tagset="universal")[0]
        stems = [stemmer.stem(token) for token in tokens]
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        word_count = sum(1 for token in tagged_tokens if token[1].isalpha())
        return stems, lemmas, tagged_tokens, word_count

    df["stem"], df["lemma"], df["word_pos"], df["word_count"] = zip(
        *df["text"].apply(preprocess_text)
    )
    df.drop(columns=["word_pos"], inplace=True)

    # Load CEFR data
    cefrj_test = cefrj.explode("headword")
    cefrj = cefrj_test[["headword", "CEFR"]]
    cefrj["stem"] = cefrj["headword"].apply(stemmer.stem)
    cefrj["lemma"] = cefrj["headword"].apply(lemmatizer.lemmatize)

    # Calculate counts
    cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    for level in cefr_levels:
        level_df = cefrj[cefrj["CEFR"] == level]
        df[f"{level}_stem_count"] = df["stem"].apply(
            lambda x: sum(1 for s in x if s in set(level_df["stem"]))
        )
        df[f"{level}_lemma_count"] = df["lemma"].apply(
            lambda x: sum(1 for s in x if s in set(level_df["lemma"]))
        )

    # Calculate POS counts
    pos_tags = ["VERB", "NOUN", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ"]
    for pos in pos_tags:
        df[f"{pos}_count"] = df["word_pos"].apply(
            lambda x: sum(1 for (_, p) in x if p == pos)
        )

    # Calculate ratios
    for pos in pos_tags:
        df[f"ratio_{pos}_word_count"] = df[f"{pos}_count"] / (df["word_count"] + 1)

    # Negative words
    with open("../../data/negative-words.txt") as f:
        negative_words = [
            line.strip() for line in f if line.strip() and line.strip()[0] != ";"
        ]
    negative_stems = set(stemmer.stem(word) for word in negative_words)
    negative_lemmas = set(lemmatizer.lemmatize(word) for word in negative_words)
    df["negative_word_lemma_count"] = df["lemma"].apply(
        lambda x: sum(1 for s in x if s in negative_lemmas)
    )
    df["negative_word_stem_count"] = df["stem"].apply(
        lambda x: sum(1 for s in x if s in negative_stems)
    )

    # Additional counts
    counts = df[
        ["word_count"]
        + [f"{level}_stem_count" for level in cefr_levels]
        + [f"{level}_lemma_count" for level in cefr_levels]
        + ["negative_word_lemma_count", "negative_word_stem_count"]
    ]

    # Finalize features
    df["cefrj_lemma_count"] = sum(df[f"{level}_lemma_count"] for level in cefr_levels)
    df["cefrj_stem_count"] = sum(df[f"{level}_stem_count"] for level in cefr_levels)

    # Final dataframe containing ratios
    df_ratio = df.filter(regex="^ratio", axis=1)
    df_ratio = pd.concat([df["id"], df_ratio], axis=1)
    df_ratio.to_csv(f"../../data/{dataset}_cefr_pos_ratios.csv", index=False)
