# Term Frequency-Inverse Document Frequency (TF-IDF) is an alternative to the word count
# calculation in a DTM
# It emphasizes important words by reducing the impact of common words

# Term Frequency
#   Problem it solves: Hight counts can dominate, especially for high frequency
#   words or long documents
# Solution: Normalize the counts so thy're all on the same scale

# Inverse Document Frequency
# Problem it solves:
#   Each word is treated equally, even when
#   some might be more important
# Solution: Assign more weight to rare words than to common words
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text_clean = pd.read_pickle('Data/text_clean.pkl')

tv = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
tfidf = tv.fit_transform(text_clean)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tv.get_feature_names_out())
print(tfidf_df)

