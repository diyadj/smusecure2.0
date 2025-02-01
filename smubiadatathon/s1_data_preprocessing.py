import pandas as pd
import re
import string
from nltk.corpus import stopwords
from collections import Counter


# Load stopwords 
STOPWORDS = set(stopwords.words('english'))

# Read CSV File
#Frontend can take in filepath, can use import os to change directory
df = pd.read_csv("data.csv", encoding='ISO-8859-1')

# Remove stopwords
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Remove punctuation
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

# Apply remove_punctuation and remove_stopwords functions, 2 new columns created
df["text_wo_punct"] = df["Text"].apply(lambda text: remove_punctuation(text))
df["text_wo_punct_stop"] = df["text_wo_punct"].apply(lambda text: remove_stopwords(text))
df.drop(["Text"], axis=1, inplace=True)

# Get the most frequent words (top 10)
cnt = Counter()
for text in df["text_wo_punct_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

# Apply remove_freqwords function to the text_wo_stop column, drop other two columns
df["text_wo_punct_stopfreq"] = df["text_wo_punct_stop"].apply(lambda text: remove_freqwords(text))
df.drop(["text_wo_punct"], axis=1, inplace=True)
df.drop(["text_wo_punct_stop"], axis=1, inplace=True)

# Get the rare words (least common words)
n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

# Apply remove_rarewords function to the text_wo_stopfreq column
df["text_wo_stopfreqrare"] = df["text_wo_punct_stopfreq"].apply(lambda text: remove_rarewords(text))
df.drop(["text_wo_punct_stopfreq"], axis=1, inplace=True)


#Remove URL
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

df["clean_text"] = df["text_wo_stopfreqrare"].apply(lambda text: remove_urls(text))
df.drop(["text_wo_stopfreqrare"], axis=1, inplace=True)

# Save the processed_data as a new csv
df.to_csv("processed_data.csv", index=True)

