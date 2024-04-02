"""Preprocess frequency lists from wikipedia article dumps.
For the English frequency file, this script creates: 
- list of top10 frequent content words
- list of top20 frequent content words
- frequency dictionary of content words 

For the fantasy and wikipedia words this script creates a list of fantasy words without the stop words.
The output is saved in json files

The full eng. freq. list contains contains 2.765.377 words and is downloaded from here: 
https://raw.githubusercontent.com/IlyaSemenov/wikipedia-word-frequency/master/results/enwiki-2023-04-13.txt

The fantasy words list contains 2.000 words and is downloaded from here:
https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Contemporary_fiction

The wikipedia words list contains 1891 words and is downloaded from here:
https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/English/Wikipedia_(2016)#1-1000

Usage:
    python preprocess_enfreqlist.py
"""
import json

from nltk.corpus import stopwords

stop_words = stopwords.words("english")

with open("../../data/word_lists/enwiki-2023-04-13.txt", "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
content_word_dict = {x[0]: x[1] for x in [line.split(" ") for line in lines]}

# Delete stop words.
keys = list(content_word_dict.keys())
for key in keys:
    if key in stop_words:
        del content_word_dict[key]

# Sort dict and get top10/top20 most frequent words.
content_sorted = dict(
    sorted(content_word_dict.items(), key=lambda item: item[1], reverse=True)
)
top_10_percent = int(len(content_sorted) * 0.1)
top10_content_words = list(content_sorted.keys())[:top_10_percent]
top_20_percent = int(len(content_sorted) * 0.2)
top20_content_words = list(content_sorted.keys())[:top_20_percent]
with open("../../data/word_lists/enfreq_data.json", "w", encoding="utf8") as f:
    json.dump([top10_content_words, top20_content_words, content_sorted], f)

with open("../../data/word_lists/fantasy_words.txt", "r") as f:
    fantasy_words = f.readlines()
fantasy_words = [word.strip() for word in fantasy_words]
fantasy_words = [word for word in fantasy_words if word not in stop_words]
with open("../../data/word_lists/fantasy_words.json", "w", encoding="utf8") as f:
    json.dump(fantasy_words, f)

with open("../../data/word_lists/wikipedia_words.txt", "r") as f:
    wikipedia_words = f.read()
wikipedia_words = [word.strip() for word in wikipedia_words]
wikipedia_words = [word for word in fantasy_words if word not in stop_words]
with open("../../data/word_lists/wikipedia_words.json", "w", encoding="utf8") as f:
    json.dump(wikipedia_words, f)
