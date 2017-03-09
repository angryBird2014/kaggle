from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import logging
from gensim import models
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier



def review_word_list(rawtext,remove_stopword=False):
    remove_html = BeautifulSoup(rawtext).get_text()
    context = re.sub('[^(a-zA-Z|0-9)]',' ',remove_html)
    words = context.lower().split()
    if remove_stopword:
        stopword_list = stopwords.words('english')
        words = [w for w in words if not w in stopword_list]
    return " ".join(words)

def review_to_sentence(review,tokenizer,remove_stopword = False):

    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for sentence in raw_sentences:
        if(len(sentence)>0):
            sentences.append(review_word_list(sentence,remove_stopword))
    return sentences

# make a review to 300 dimension
def makeFeatureVector(words,model,numfeatures):

    featureVec = np.zeros([num_features,],dtype=np.float32)

    #index2word = set(model.index2word)

    index2word = list(model.wv.index2word)
    for word in words:
        if word in index2word:
            word_vector = model[word]
            #print(word,'------>',word_vector)
            featureVec = np.add(featureVec,word_vector)
    featureVec = np.divide(featureVec,len(words))
    return featureVec

#get all review to a matrix
def getAverageVector(reviews,model,num_features):

    aveReviewVector = np.zeros([len(reviews),num_features],dtype=np.float32)
    count = 0
    for review in reviews:

        featureVector = makeFeatureVector(reviews,model,num_features)
        aveReviewVector[count] = featureVector
        count += 1
    return aveReviewVector

if __name__ == '__main__':

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    train_data = pd.read_csv('4linelabelTrainData.tsv', header=0, delimiter="\t", quoting=3)

    test_data = pd.read_csv('testData.tsv', header=0, delimiter="\t")
    unlabel_data = pd.read_csv('4lineUnlabelTrainData.tsv', header=0, delimiter="\t", quoting=3)
    print(train_data['review'].size, test_data['review'].size, unlabel_data['review'].size)

    sentences = []
    for review in train_data['review']:
        sentences += review_to_sentence(review,tokenizer)
    for review in unlabel_data['review']:
        sentences += review_to_sentence(review,tokenizer)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 5 # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    model_name = "300features_40minwords_10context"
    sentences = [s.split() for s in sentences]
    if os.path.exists(os.path.join(model_name)):
        #model = word2vec.Word2Vec.load(model_name)
        model = models.Word2Vec.load(model_name)
    else:
        print("Training model...")
        model = models.Word2Vec(sentences, workers=num_workers,size=num_features, min_count=min_word_count,window=context, sample=downsampling)
        model.init_sims(replace=True)
        model.save(model_name)
    print(model.wv.vocab)
    #print(model['cinema'])
    clean_train_reviews = []
    for train_view in train_data['review']:
        clean_train_reviews.append(review_word_list(train_view,remove_stopword=True))
    clean_test_view = []
    for test_view in test_data['review']:
        clean_test_view.append(review_word_list(test_view,remove_stopword=True))
    train_vector = getAverageVector(clean_train_reviews,model,num_features)
    test_vector = getAverageVector(clean_test_view,model,num_features)
    forest = RandomForestClassifier(n_estimators=10)
    classify_estimator = forest.fit(train_vector,train_data['sentiment'])
    result = classify_estimator.predict(test_vector)
    output = pd.DataFrame(data={"id": test_data["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
