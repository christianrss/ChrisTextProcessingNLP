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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text_clean = pd.read_pickle('Data/text_clean.pkl')

tv = TfidfVectorizer(stop_words='english', min_df=0.1, max_df=.5)
tfidf = tv.fit_transform(text_clean)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tv.get_feature_names_out())

top_weights = tfidf_df.sum().sort_values().tail(10)
top_weights.plot(kind='barh')
plt.savefig('top_weights.png')

cv = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=.2, max_df=.8)
dtm = cv.fit_transform(text_clean)
dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())

term_freq = dtm_df.sum().sort_values().tail(10)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
term_freq.plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Count Vectorizer')

top_weights.plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title('TF-IDF Vectorizer')

plt.tight_layout()
plt.savefig('analysis.png')