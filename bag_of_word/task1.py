import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words(rawtext):
    context = BeautifulSoup(rawtext, 'lxml')
    letters_context = re.sub('[^a-zA-Z]', " ", context.get_text())
    lower_context = letters_context.lower()
    words = lower_context.split()
    words_except_stop_words = [w for w in words if not w in stopwords]
    return " ".join(words_except_stop_words)




if __name__ == '__main__':
    nltk.download()
    train_data = pd.read_csv('labeledTrainData.tsv',delimiter='\t',quoting=3)
    test_data = pd.read_csv('testData.tsv',delimiter='\t',quoting=3)
    clean_test_view = []
    clean_train_view = []
    for i in train_data['review'].size:
        clean_train_view.append(review_to_words(train_data['review'][i]))
    for i in test_data['review'].size:
        clean_test_view.append(review_to_words(test_data['review'][i]))
    vectorizer = CountVectorizer(analyzer="word",tokenizer = None,preprocessor=None,stop_words=None,max_features=5000)
    train_feature = vectorizer.fit_transform(clean_train_view)
    train_feature = np.array(train_feature)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_feature,train_data['sentiment'])
    result = model.predict(clean_test_view)
    output = pd.DataFrame(data={'id':test_data['id'],'sentiment':result})
    output.to_csv("bag_of_word_model.csv",index=False,quoting=3)
