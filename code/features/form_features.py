"""Compute emotion, sentiment, formality and toxicity features using pretrained transformer models.
Select the dataset to copmute the measures for in line 15. 

Usage:
    python form_features.py
"""

import torch
import pandas as pd
from transformers import pipeline
from datasets import load_dataset


if __name__ == "__main__":
    dataset = "test"
    data = pd.read_json(
        path_or_buf=f"../../data/subtaskA_{dataset}_monolingual.jsonl",
        lines=True,
        engine="pyarrow",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Emotion analysis.
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=device,
    )
    predictions = classifier(
        data["text"],
        truncation=True,
        padding=True,
        batch_size=100,
        return_all_scores=True,
    )

    # Sentiment analysis.
    sent_classifier = pipeline(
        "sentiment-analysis", device=device, return_all_scores=True
    )
    sent_predictions = sent_classifier(
        data["text"],
        truncation=True,
        padding=True,
        batch_size=100,
        return_all_scores=True,
    )

    # Formality analysis.
    form_classifier = pipeline(
        "text-classification",
        model="s-nlp/roberta-base-formality-ranker",
        device=device,
        return_all_scores=True,
    )
    form_predictions = form_classifier(
        data["text"],
        truncation=True,
        padding=True,
        batch_size=100,
        return_all_scores=True,
    )

    # Toxicity analysis.
    toxic_classifier = pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=0,
        return_all_scores=True,
    )
    toxic_predictions = toxic_classifier(
        data["text"],
        truncation=True,
        padding=True,
        batch_size=100,
        return_all_scores=True,
    )

    # Save the features.
    features = pd.DataFrame(
        {
            "emotion": predictions,
            "sentiment": sent_predictions,
            "formality": form_predictions,
            "toxicity": toxic_predictions,
        }
    )
    features.to_pickle(f"../../data/feature_metrics/{dataset}A_form_metrics.pkl")
