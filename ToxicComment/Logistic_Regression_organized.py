print("Hello World!")

import numpy as np
import pandas as pd

import operator
import time
import sys 

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_union
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

sys.setrecursionlimit(5000)

def read_data():

    start_time = time.time()
    #train = pd.read_csv("Data/" + "train_preprocessed.csv")
    #test = pd.read_csv("Data/" + "test_preprocessed.csv")
        #roc_auc: 0.9854526885766424
        #miniscule improvement
    
    train = pd.read_csv("Data/" + "train.csv").fillna(" ")
    test = pd.read_csv("Data/" + "test.csv").fillna(" ")
        #roc_auc: 0.9854526365101194
    
    end_time = time.time()
    
    print("Data read.")
    print("Data reading time:", end_time - start_time)

    #print("train shape:", train.shape)    # (159571, 8)
    #print("test shape:", test.shape)      # (153164, 2)
    
    #print("train columns:", train.columns)
        # 'id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    #print("test columns:", test.columns)
        # 'id', 'comment_text'
        
    return train, test

stop = stopwords.words("english")

porter = PorterStemmer()

def tokenizer(text):
    return [word for word in text.split()]

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]   #argsort returns the indices that would sort an array
    print("topn_ids:", topn_ids)
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def make_vectorizer():
    word_vectorizer = TfidfVectorizer(sublinear_tf = True, 
                                      strip_accents = "unicode", 
                                      analyzer = "word",
                                      token_pattern = r'\w{1,}',
                                      ngram_range = (1,1),
                                      )
                                        #max_features = 30000,
                                      
    
    #use a stripped version for gridsearch
#    word_vectorizer = TfidfVectorizer(sublinear_tf = True, 
#                                      strip_accents = "unicode", 
#                                      analyzer = "word",
#                                      )
    
    
        #ngram_range = (1,1) gives 97% score. How about (1,2)?
            #doesnt help. May be slightly worse.
        #not limiting max_features didn't help or hurt.
        #not using idf did hurt the perfoermance a bit. Also common words like the, i, and
            #started getting the top scores. Expected.
        
    char_vectorizer = TfidfVectorizer(sublinear_tf = True,
                                      strip_accents = "unicode",
                                      analyzer = "char",
                                      ngram_range = (1,4),
                                      )
                                    #max_features = 30000
    
    #using char_vectorizer imporved perforance.
    #reducing max_features to 10K didn't have a noticeable perforance lag.
    
    #vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs = 2)
    vectorizer = FeatureUnion( transformer_list = [("wv", word_vectorizer), ("cv", char_vectorizer)], n_jobs = 2)
    
    #vectorizer = word_vectorizer

    return vectorizer

    pass


def play_with_tfidf_features(vectorizer, all_text):
    '''
    Some testing and playing to better understand tfidf feature. Nothing important.
    
    Here, the assumption is that vectorizer is a feature union of word vectorizer and
    char vectorizer.
    
    If the word vectorizer is directly passed, just remove transformer_list[0][1] 
    attribute while selecting the tfidf object.
    '''
    start_time = time.time()
    
    vectorizer.fit(all_text) #this fitting seems unnecessary as we are later fitting the vectorizer again inside the pipeline
    end_time = time.time()
    print("word vectorizer fitting time: ", end_time - start_time)
    
    print("vocabulary size: ", len(vectorizer.transformer_list[0][1].vocabulary_))
        #here transformer_list is a list of tuples of (transformer_name, transformer_obejct) format.
        #so transformer_list[0][1] means get the second element of the first transformer tuple, which in our case is the word_vectorizer object
        
    #print("vocabulary size: ", len(vectorizer.vocabulary_))
    print("first few:")
    print(sorted(vectorizer.transformer_list[0][1].vocabulary_.items(), key = operator.itemgetter(1), reverse = True)[:10])
    print(vectorizer.transformer_list[0][1].get_feature_names()[:10])
    #print(vectorizer.get_feature_names()[:10])
    
    start_time = time.time()
    train_features = vectorizer.transform(train_text)
    end_time = time.time()
    print("train_text transformation time: ", end_time - start_time)
    
    print("first train sample:")
    print(type(train_features[0]))
    print(sorted(np.squeeze(train_features[0].toarray()))[::-1][:10])
    
    row = np.squeeze(train_features[0].toarray())
    print("row[0]:",row)
    features = vectorizer.transformer_list[0][1].get_feature_names()
    #features = vectorizer.get_feature_names()
    top_tf = top_tfidf_feats(row, features, 25)
    print(top_tf)
    
    pass


def perform_gridsearch(train, test, class_names, vectorizer):
    
    #lets play with gridsearch
    #    param_grid = [ { 'vect__ngram_range': [(1,1),(2,2),(1,2)],
    #                 'vect__stop_words': [stop, None],
    #                 'vect__tokenizer': [tokenizer,tokenizer_porter],
    #                 'vect__max_features': [30000],
    #                 'clf__C': [1.0, 10.0, 100.0]
    #                 },
    #                { 'vect__ngram_range': [(1,1),(2,2),(1,2)],
    #                 'vect__stop_words': [stop, None],
    #                 'vect__tokenizer': [tokenizer,tokenizer_porter],
    #                 'vect__use_idf': [False],
    #                 'clf__C': [1.0, 10.0, 100.0]
    #                 },
    #                ]
    #

    param_grid = [ { 
                 'vect__wv__max_features': [3000, 1500],
                 'vect__cv__max_features': [3000, 1500],
                 'clf__C': [1.0, 10.0, 100.0]
                 },
                ]
    

    scores = []
    best_parameters = []
    
    train_text = train["comment_text"]
    test_text = test["comment_text"]
    
    #submission = pd.DataFrame.from_dict({"id": test["id"]})
    
    for target_class in class_names:
    
        print("\nWorking with target_class: ", target_class)
        
        train_target = train[target_class]
        
        classifier = LogisticRegression(solver = "sag", n_jobs = 4)
        
        lr_vectorizer = Pipeline( [ ('vect', vectorizer), ('clf', classifier) ] )
        
        #change scoring to accuracy for testing
        #But for real run, roc_auc is better
        gs_lr_vectorizer = GridSearchCV(lr_vectorizer, param_grid, 
                                        scoring = 'accuracy', cv = 5, 
                                        verbose = 1, n_jobs = 4)
                
        #fit the grid here
        #get the best results
        
        start_time = time.time()
        
        gs_lr_vectorizer.fit(train_text, train_target)
        
        
        end_time = time.time()
        print("Grid search time:", end_time - start_time)
        
        best_params = gs_lr_vectorizer.best_params_
        best_parameters.append(best_params)
        
        print("best parameter set:", best_params)
        
        cv_score = gs_lr_vectorizer.best_score_
        scores.append(cv_score)
        print("cv_score for class {} : {}".format(target_class, cv_score))

        #submission[target_class] = gs_lr_vectorizer.predict_proba(test_text)[:,1]

    #submission.to_csv("submission_gs.csv", index = False)

    return best_parameters, scores
    
    #best parameters found by grid search
    #toxic:
    #best parameter set: {'clf__C': 10.0, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>, 'vect__use_idf': False}
    #severe_toxic:
    #best parameter set: {'clf__C': 1.0, 'vect__max_features': 30000, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>}
    #obscene:
    #{'clf__C': 10.0, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>, 'vect__use_idf': False}
    #threat:
    #best parameter set: {'clf__C': 10.0, 'vect__max_features': 30000, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>}
    #insult:
    #best parameter set: {'clf__C': 1.0, 'vect__max_features': 30000, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>}
    #identity_hate:
    #best parameter set: {'clf__C': 1.0, 'vect__max_features': 30000, 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer_porter at 0x7f0690070620>}

    pass

def gridsearch_with_different_classifiers(train, test, class_names, vectorizer):
    '''
    Ideally, in this step we won't bother about the vectorizers as we already 
    did extensive grid search with them.
    
    Here we will probe with different classifiers and their hyper-parameters.
    We can have logistic regression, svm, and something else (what?).
    
    Then once we have the best parameters, we can have a voting classifier with 
    the classifiers with their best hyper-parameters.
    
    '''
    #lets try a simple grid search with svm
    
    '''
    IMPORTANT: lesson learned. SVM is too slow. Not feasible for a dataset
                larger than, say, 10K samples. Abandon.
    '''
    
    param_grid = [ 
                    { 'clf__C': [0.1, 1.0, 10.0],
                      'clf__kernel' : ['linear']   
                     },
                    {  'clf__C' : [0.1, 1.0, 10.0],
                       'clf__kernel' : ['rbf'],
                       'clf__gamma' : [0.1, 1.0, 10.0]
                       }
                ]
    
    scores = []
    best_parameters = []
    
    train_text = train["comment_text"]
    test_text = test["comment_text"]
    
    submission = pd.DataFrame.from_dict({"id": test["id"]})
    
    for target_class in class_names:
    
        print("\nWorking with target_class: ", target_class)
        
        train_target = train[target_class]
        
        classifier = SVC(random_state = 42)
        
        lr_vectorizer = Pipeline( [ ('vect', vectorizer), ('clf', classifier) ] )
        
        #change scoring to accuracy for testing
        #But for real run, roc_auc is better
        gs_lr_vectorizer = GridSearchCV(lr_vectorizer, param_grid, 
                                        scoring = 'accuracy', cv = 5, 
                                        verbose = 1, n_jobs = 4)
                
        #fit the grid here
        #get the best results
        
        start_time = time.time()
        
        gs_lr_vectorizer.fit(train_text, train_target)
        
        end_time = time.time()
        print("Grid search time:", end_time - start_time)
        
        best_params = gs_lr_vectorizer.best_params_
        best_parameters.append(best_params)
        
        print("best parameter set:", best_params)
        
        cv_score = gs_lr_vectorizer.best_score_
        scores.append(cv_score)
        print("cv_score for class {} : {}".format(target_class, cv_score))

        submission[target_class] = gs_lr_vectorizer.predict_proba(test_text)[:,1]

    submission.to_csv("submission_gs_svm.csv", index = False)

    
    pass


def predict_with_best_parameters(train, test, class_names):
    
    submission = pd.DataFrame.from_dict({"id": test["id"]})
    
    scores = []
    
    #assuming best parameters are already found by grid_search
    best_parameters = {
            "toxic" : {'clf': {'C': 10.0},
                       'vect' : {'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter, 'use_idf': False},
                       },
            "severe_toxic": {'clf':  {'C': 1.0},
                             'vect': {'max_features': 30000, 'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter},
                     },
            "obscene": {'clf': {'C': 10.0}, 
                        'vect': {'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter, 'use_idf': False},
                        },
            "threat": {'clf' : {'C': 10.0}, 
                       'vect' : {'max_features': 30000, 'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter},
                       },
            "insult": {'clf' : {'C': 1.0}, 
                       'vect' : {'max_features': 30000, 'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter},
                       },
            "identity_hate" : {'clf': {'C': 1.0}, 
                                 'vect' : {'max_features': 30000, 'ngram_range': (1, 1), 'stop_words': None, 'tokenizer': tokenizer_porter}
                      }
            }

    for target_class in class_names:
    
        print("\nWorking with target_class: ", target_class)
        
        train_target = train[target_class]
        
        word_vectorizer = TfidfVectorizer(sublinear_tf = True, 
                                      strip_accents = "unicode", 
                                      analyzer = "word",
                                      **best_parameters[target_class]["vect"]
                                      )
        char_vectorizer = TfidfVectorizer(sublinear_tf = True,
                                      strip_accents = "unicode",
                                      analyzer = "char",
                                      ngram_range = (1,4),
                                      max_features = 30000)
        vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs = 2)
        
        classifier = LogisticRegression(solver = "sag", n_jobs = 4, **best_parameters[target_class]["clf"])
    
        lr_vectorizer = Pipeline( [ ('vect', vectorizer), ('clf', classifier) ] )
        
        
        start_time = time.time()
        
        cv_score = np.mean(cross_val_score(lr_vectorizer, train_text, train_target, cv = 3, scoring = "roc_auc", n_jobs = 3))
        
        end_time = time.time()
        print("CV time:", end_time - start_time)
        
        scores.append(cv_score)
        print("cv_score for class {} : {}".format(target_class, cv_score))
        
        start_time = time.time() 
        lr_vectorizer.fit(train_text, train_target)
        end_time = time.time()
        print("fitting time: ", end_time - start_time)
        
        start_time = time.time()
        submission[target_class] = lr_vectorizer.predict_proba(test_text)[:,1]
            #predict_proba returns two columns. The first is the probability that
            # the sample is of class 0, the second is the probability that the 
            # sample is of class 1. So we only need to slice the second column.
        end_time = time.time()
        print("Prediction time: ", end_time - start_time)
        
    print("total CV score is: {}".format(np.mean(scores)))    
    
    submission.to_csv("submission_gs_best_est_union_and_piped.csv", index = False)

    pass


if __name__ == "__main__":

    train, test = read_data()
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    #class_names = ['toxic']
    
    train_text = train["comment_text"]
    print(train_text.shape)
    print(type(train_text))
        # <class 'pandas.core.series.Series'>
        # (159571,)
    
    test_text = test["comment_text"]
    print(test_text.shape)
    print(type(test_text))
        # <class 'pandas.core.series.Series'>
        # (153164,)
    
    all_text = pd.concat([train_text, test_text])
    print(all_text.shape)
    print(type(all_text))
        # <class 'pandas.core.series.Series'>
        # (312735,)
    
    vectorizer = make_vectorizer()
    
    #play_with_tfidf_features(vectorizer, all_text)
       
    #best_parameters, scores = perform_gridsearch(train, test, class_names, vectorizer)
    
    gridsearch_with_different_classifiers(train, test, class_names, vectorizer)
    
    #predict_with_best_parameters(train, test, class_names)
    
    print("The world never says hello back!")