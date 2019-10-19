import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle
import time
import os
import os.path
import sys

sys.setrecursionlimit(5000)

pickle_location = "PickleJars"
os.makedirs(pickle_location, exist_ok = True)


start_time = time.time()
df_train = pd.read_csv("Data/" + "train_preprocessed.csv")
df_test = pd.read_csv("Data/" + "test_preprocessed.csv")
end_time = time.time()

print("Data read.")
print("Data reading time:", end_time - start_time)
print("train shape:", df_train.shape)
print("test shape:", df_test.shape)

#There is one entry in df_train for which processed_text is empty.
#Correct that.

#df_train[df_train["processed_text"].isnull()]
#the location is row 127051

df_train.at[127051, "processed_text"] = "try reviewing it again will try to improve on this article"
#print(df_train.iloc[127051])

#shuffle the training data before splitting into train-test portions
#df_train = df_train.sample(frac = 1).reset_index(drop = True)

#tokenize/stop-word removal
def tokenizer(text):
    return text.split()

print(tokenizer("Runners like running and thus they run."))

#porter-stemmer tokenizer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter("Runners like running and thus they run."))

stop = stopwords.words("english")

print([w for w in tokenizer_porter("a runner likes running and runs a lot") if w not in stop])

X = df_train["processed_text"]
# just select one column for the time being and try to concentrate on that
cols = ["toxic"]
y_toxic = df_train[cols]

cols = ["obscene"]
y_obscene = df_train[cols]

#convert it to numpy array
y_toxic = y_toxic.values.ravel()

y_obscene = y_obscene.values.ravel()

#print(y_toxic.shape)    #(159571,)
#print(type(y_toxic))    #<class 'numpy.ndarray'>
X_train, X_test, y_toxic_train, y_toxic_test = train_test_split(X, y_toxic, test_size = 0.20, random_state = 42 )

#X_train, X_test, y_obscene_train, y_obscene_test = train_test_split(X, y_toxic, test_size = 0.20, random_state = 42 )

#print(X_train.shape, X_test.shape, y_toxic_train.shape, y_toxic_test.shape)
    # (127656,) (31915,) (127656,) (31915,)

pickle_name = "tfidf.pk"
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    #train a tfidf vectorizer first and use it for the rest of the classifiers
    tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, preprocessor = None, ngram_range = (1,1), stop_words = stop, tokenizer = tokenizer_porter, use_idf = True) 
    
    tfidf = tfidf.fit(X_train)

    end_time = time.time()
    print("Tfidf fitting time:", end_time - start_time)

    pickle.dump(tfidf, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    tfidf = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("tfidf pickle loaded.")

pickle_name = "X_train_tfidf.pk"
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    #train a tfidf vectorizer first and use it for the rest of the classifiers
    X_train_tfidf = tfidf.transform(X_train)

    end_time = time.time()
    print("Tfidf transform time for train:", end_time - start_time)

    pickle.dump(X_train_tfidf, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    X_train_tfidf = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("X_train_tfidf pickle loaded.")

pickle_name = "X_test_tfidf.pk"
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    #train a tfidf vectorizer first and use it for the rest of the classifiers
    X_test_tfidf = tfidf.transform(X_test)

    end_time = time.time()
    print("Tfidf transform time for test:", end_time - start_time)

    pickle.dump(X_test_tfidf, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    X_test_tfidf = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("X_test_tfidf pickle loaded.")

#feature_names = tfidf.get_feature_names()


# do regression with tfidf
pickle_name = "regression.pk"
print("Logistic regression: ")
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    lr = LogisticRegression(random_state = 0, penalty = "l2", C = 10.0 )
    lr = lr.fit(X_train_tfidf, y_toxic_train)
    end_time = time.time()
    print("Logistic regression training time:", end_time - start_time)

    pickle.dump(lr, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    lr = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("Logistic regression pickle loaded.")

lr_score_train = lr.score(X_train_tfidf, y_toxic_train)
lr_score_test = lr.score(X_test_tfidf, y_toxic_test)

print("Accuracy score, train: {:.2f}".format(lr_score_train))
print("Accuracy score, test: {:.2f}".format(lr_score_test))

##now try perceptron
pickle_name = "perceptron.pk"
print("Perceptron: ")
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    pt = Perceptron(random_state = 0, class_weight = "balanced", max_iter = 1000, tol = 1e-3)
    pt = pt.fit(X_train_tfidf, y_toxic_train)
    end_time = time.time()
    print("Perceptron training time:", end_time - start_time)
    
    pickle.dump(pt, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    pt = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("Perceptron pickle loaded.")

pt_score_train = pt.score(X_train_tfidf, y_toxic_train)
pt_score_test = pt.score(X_test_tfidf, y_toxic_test)

print("Accuracy score, train: {:.2f}".format(pt_score_train))
print("Accuracy score, test: {:.2f}".format(pt_score_test))


## 'Support Vector Machines'
pickle_name = "svm.pk"
print("SVM: ")
if os.path.isfile(os.path.join(pickle_location, pickle_name)) is False:    
    start_time = time.time()
    #svm = SVC(random_state = 42, class_weight = "balanced", max_iter = 1000, tol = 1e-3)
    svm = SVC(random_state = 42)
    svm = svm.fit(X_train_tfidf, y_toxic_train)
    end_time = time.time()
    print("SVM training time:", end_time - start_time)
    
    pickle.dump(svm, open(os.path.join(pickle_location, pickle_name),"wb"))

else:
    svm = pickle.load(open(os.path.join(pickle_location, pickle_name),"rb"))
    print("SVM pickle loaded.")

svm_score_train = svm.score(X_train_tfidf, y_toxic_train)
svm_score_test = svm.score(X_test_tfidf, y_toxic_test)

print("Accuracy score, train: {:.2f}".format(svm_score_train))
print("Accuracy score, test: {:.2f}".format(svm_score_test))



    
'''
Takes extremely lng, and eventually crashes showing memory error. Better to skip
for now.
'''
# now try KNN
#start_time = time.time()
#knn = KNeighborsClassifier()
#knn = knn.fit(X_train_tfidf, y_toxic_train)
#end_time = time.time()
#
#knn_score_train = knn.score(X_train_tfidf, y_toxic_train)
#knn_score_test = pt.score(X_test_tfidf, y_toxic_test)
#print("KNN: ")
#print("KNN time:", end_time - start_time)
#print("Accuracy score, train: {:.2f}".format(knn_score_train))
#print("Accuracy score, test: {:.2f}".format(knn_score_test))

