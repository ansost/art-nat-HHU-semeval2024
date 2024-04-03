import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt


# Function to load dataset from JSON files
def load_json_dataset(datapath):
    dataset = load_dataset("json", data_files=datapath)
    return dataset["train"]


# Function to compute frequency features
def compute_frequency_features(text, fdist_filtered_log, fdist):
    text_tokens = word_tokenize(text)
    freqs = [
        fdist_filtered_log[w] for w in text_tokens if w in fdist_filtered_log.keys()
    ]
    num_freq1 = sum(1 for w in text_tokens if fdist[w] == 1)
    prop_freq_content = sum(1 for f in freqs if f <= low) / max(len(freqs), 1)
    prop_unfreq_content = sum(1 for f in freqs if f >= high) / max(len(freqs), 1)
    mean_log_freq_content = np.mean(freqs) if freqs else 0
    return mean_log_freq_content, num_freq1, prop_freq_content, prop_unfreq_content


# Function to scale features using MinMaxScaler
def scale_features(df):
    scaler = MinMaxScaler()
    for feat in [
        "mean_log_freq_content_words",
        "num_freq1_words",
        "prop_freq_content_words",
        "prop_unfreq_content_words",
    ]:
        scaler.fit(df[feat].values.reshape(-1, 1))
        scaled_values = scaler.transform(df[feat].values.reshape(-1, 1))
        df[feat] = scaled_values.flatten()
    return df


# Function to compute correlations between features and labels
def compute_correlations(df):
    correlations = {}
    for feat in [
        "mean_log_freq_content_words",
        "num_freq1_words",
        "prop_freq_content_words",
        "prop_unfreq_content_words",
    ]:
        correlations[feat] = {
            "spearmanr": spearmanr(df[feat], df["label"]),
            "pearsonr": pearsonr(df[feat], df["label"]),
        }
    return correlations


# Function to train Logistic Regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    coefficients = clf.coef_
    intercept = clf.intercept_
    classes = clf.classes_
    return accuracy, f1, coefficients, intercept, classes


if __name__ == "__main__":
    datapath_train = "subtaskA_train_monolingual.jsonl"
    datapath_dev = "subtaskA_dev_monolingual.jsonl"
    ds_train = load_json_dataset(datapath_train)
    ds_dev = load_json_dataset(datapath_dev)

    # Filter human examples
    ds_human_train = ds_train.filter(lambda example: example["model"] == "human")
    ds_human_dev = ds_dev.filter(lambda example: example["model"] == "human")

    # Tokenize and compute frequency distribution
    tokens_train = [
        word for example in ds_human_train["text"] for word in word_tokenize(example)
    ]
    tokens_dev = [
        word for example in ds_human_dev["text"] for word in word_tokenize(example)
    ]

    fdist_train = FreqDist(tokens_train)
    fdist_dev = FreqDist(tokens_dev)

    # Compute filtered frequency distribution
    stop_words = set(stopwords.words("english"))
    punctuation = list(".,,.,;:!?()[]{}`''\"@#$^&*+-|=~_ ")
    stop_words.update(punctuation)
    filtered_tokens_train = [
        w.lower() for w in tokens_train if not w.lower() in stop_words
    ]
    filtered_tokens_dev = [w.lower() for w in tokens_dev if not w.lower() in stop_words]

    fdist_filtered_train = FreqDist(filtered_tokens_train)
    fdist_filtered_dev = FreqDist(filtered_tokens_dev)

    # Compute percentiles
    sorted_values_train = sorted(fdist_filtered_train.values())
    l_train = int(round(len(sorted_values_train) * 0.1, 0))
    h_train = int(round(len(sorted_values_train) * 0.9, 0))
    low_train = sorted_values_train[l_train]
    high_train = sorted_values_train[h_train]

    # Compute frequency features
    df_train = pd.DataFrame(ds_train)
    (
        df_train["mean_log_freq_content_words"],
        df_train["num_freq1_words"],
        df_train["prop_freq_content_words"],
        df_train["prop_unfreq_content_words"],
    ) = zip(
        *df_train["text"].apply(
            lambda text: compute_frequency_features(
                text, fdist_filtered_train, fdist_train
            )
        )
    )
    df_dev = pd.DataFrame(ds_dev)
    (
        df_dev["mean_log_freq_content_words"],
        df_dev["num_freq1_words"],
        df_dev["prop_freq_content_words"],
        df_dev["prop_unfreq_content_words"],
    ) = zip(
        *df_dev["text"].apply(
            lambda text: compute_frequency_features(text, fdist_filtered_dev, fdist_dev)
        )
    
    # save
    df_train.to_csv("train_freq_features.csv", index=False)
    df_dev.to_csv("dev_freq_features.csv", index=False)