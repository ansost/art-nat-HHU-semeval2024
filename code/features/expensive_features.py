"""Compute spacy-based features for the training and dev dataset.
Features:
    - Mean dependency distance per text
    - Maximum distane in a tree
    - Verb-noun ratio
    - Count negation words from manual list (no, not, never, none, nobody, nothing, neither)
    - Average passive construction count per sentence

INPUT must be either the training, dev or test data for task A (subtaskA_train_monolingual.jsonl, subtaskA_dev_monolingual.jsonl, or test data).
Change line 36 according to which dataset you want to use by specifying either 'train' 'dev', or 'test'.

OUTPUT concatenated with the input data is saved as a hdf5 file. 
It is reccommended to load the output file using: 'df = pd.read_hdf("../data/trainA_cheap_metrics.h5", "df")'.

Usage: 
    python computeSpacyFeatures.py
"""
import spacy
import pandas as pd
from tqdm import tqdm


def walk_tree(node, depth):
    """Compute the depth of each subtree.
    More specifically, compute the depth of each word in the sentence,
    where the depth is the distance from the root.
    """
    depths[node.orth_] = depth
    if node.n_lefts + node.n_rights > 0:
        return [walk_tree(child, depth + 1) for child in node.children]


def valid_dataset(dataset):
    """Check if dataset is valid."""
    if dataset not in ["dev", "train", "test"]:
        raise ValueError("Dataset must be one of: dev, train, test")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    dataset = "train"
    valid_dataset(dataset)

    df = pd.read_json(
        path_or_buf=f"../../data/subtaskA_{dataset}_monolingual.jsonl",
        lines=True,
        engine="pyarrow",
    )
    texts = df["text"].to_list()

    depths = {}
    for text in tqdm(texts):
        doc = nlp(text)
        # Max and mean dependency distance per text.
        for sent in doc.sents:
            [walk_tree(sent.root, 0) for sent in doc.sents]
            max_depth = max(depths.values())
            mean_depth = sum(depths.values()) / len(depths)
        # Verb-noun ratio per text.
        n_verbs = len([token for token in doc if token.pos_ == "VERB"])
        n_nouns = len([token for token in doc if token.pos_ == "NOUN"])
        verb_noun_ratio = n_verbs / n_nouns if n_nouns != 0 else 0
        # Negation words (manual list) per words.
        negation_words = ["no", "not", "never", "none", "nobody", "nothing", "neither"]
        n_negation_words = len([token for token in doc if token.text in negation_words])
        n_words = len([token for token in doc])
        n_negation_words = n_negation_words / n_words if n_words != 0 else 0
        # Average passive construction count per sentence.
        n_passive_constructions = len(
            [token for token in doc if token.dep_ == "nsubjpass"]
        )
        avg_passive_constructions = (
            n_passive_constructions / len(list(doc.sents))
            if len(list(doc.sents)) != 0
            else 0
        )
        # Save features to dataframe.
        df.loc[df["text"] == text, "max_depth"] = max_depth
        df.loc[df["text"] == text, "mean_depth"] = mean_depth
        df.loc[df["text"] == text, "verb_noun_ratio"] = verb_noun_ratio
        df.loc[df["text"] == text, "n_negation_words"] = n_negation_words
        df.loc[
            df["text"] == text, "avg_passive_constructions"
        ] = avg_passive_constructions
    # print(df[df.columns[-5:]].head())
    df.to_hdf(
        f"../../data/feature_metrics/{dataset}A_expensive_metrics.h5",
        key="df",
        mode="w",
    )
