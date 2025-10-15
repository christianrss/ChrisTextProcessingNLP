import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

series = pd.read_pickle('Data/text_clean.pkl')

cv = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=.2, max_df=.8)
dtm = cv.fit_transform(series)
dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())

term_freq = dtm_df.sum()
term_freq.sort_values().plot(kind='barh')
plt.savefig('chart.png')