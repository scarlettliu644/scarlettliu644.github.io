import pandas as pd # in order to process the excel, tool Pandas is needed
df = pd.read_csv("data-national-organised-all.csv", encoding='utf-8') # read csv
df.head() # show data

import jieba
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
df["content_cutted"] = df.Topic.apply(chinese_word_cut)
df.content_cutted.head() # show data 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
n_features = 500 # set the number of key words
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(df.content_cutted)

from sklearn.decomposition import LatentDirichletAllocation
n_topics = 7 # set the number of topics for selected key words
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    
n_top_words = 15 # say there are 5 key words for each topic
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words) # show 

