import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

series = pd.read_pickle('Data/text_clean.pkl')

cv = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=.2, max_df=.8)
dtm = cv.fit_transform(series)
dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())
print(dtm_df)