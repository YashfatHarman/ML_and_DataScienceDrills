import numpy as np
import pandas as pd

import time
import pickle
import os
import operator

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_union
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin

# we already preprocessed the data. Made custom features. Cleaned the comment texts.
# Decided not to use leaky features for the time being.

# Now let's start making the models. First: direct features
    # TF-IDF (could use CountVectorizer or Hashing Vectorizer as well.)
    

# read data
def read_data():
    start_time = time.time()
    
    train = pd.read_csv("../Data/train_feats_clean.csv")
    test = pd.read_csv("../Data/test_feats_clean.csv")
    
    #TODO: read_csv() may have been causing some error by auto-converting some
        #text value to float. Later in tfidf we are getting errors like
        # AttributeError: 'float' object has no attribute 'lower'.
        #One possible direction may be to forcefully ensure that text columns
        #are read as text. Find out the best practice.
    
    train["clean_text"] = train["clean_text"].apply(lambda x: str(x))    
    test["clean_text"] = test["clean_text"].apply(lambda x: str(x))    
    
    #    print("train shape:", train.shape)
    #    print("train columns:", train.columns)
        #        ['id', 'comment_text', 'count_sent', 'count_word', 'count_unique_word',
        #       'count_letters', 'count_punctuations', 'count_words_upper',
        #       'count_words_title', 'count_stopwords', 'mean_word_len',
        #       'word_unique_percent', 'punct_percent', 'spam', 'toxic', 'severe_toxic',
        #       'obscene', 'threat', 'insult', 'identity_hate', 'clean', 'clean_text']
    
    #    print("test shape:", test.shape)
    #    print("test columns:", test.columns)
        #       ['id', 'comment_text', 'count_sent', 'count_word', 'count_unique_word',
        #       'count_letters', 'count_punctuations', 'count_words_upper',
        #       'count_words_title', 'count_stopwords', 'mean_word_len',
        #       'word_unique_percent', 'punct_percent', 'spam', 'clean_text']
            
    train_unlabeled = pd.concat([train.iloc[:,0:14], train["clean_text"]], axis=1)
                    # Slice the columns from id to spam, and then concat clean_text.
                    # The same ones as in Test.
                    
    #    print(train_unlabeled.shape)
    #    print("train_unlabeed columns:", train_unlabeled.columns)
    #    print("sample train_unlabeled:", train_unlabeled.iloc[0,:])
    #    print("comment text:", train_unlabeled.iloc[0]["comment_text"])
    #    print("clean text:", train_unlabeled.iloc[0]["clean_text"])

    merged = pd.concat([train_unlabeled, test], ignore_index = True)
    
    #    print("\n\n\n")
    #    print(merged.shape)
    #    print("merged columns:", merged.columns)
    #    print("sample merged:", merged.iloc[0,:])
    #    print("comment text:", merged.iloc[0]["comment_text"])
    #    print("clean text:", merged.iloc[0]["clean_text"])

    end_time = time.time()
    print("data reading time:", end_time - start_time)
    
    return train, test, merged

def read_data_and_query():
    '''
    For quick and dirty query over raw data
    '''
    train = pd.read_csv("../Data/train_feats_clean.csv")
    test = pd.read_csv("../Data/test_feats_clean.csv")
    
    
    #get the first row that is obscene
    obscene_row = train[ train["obscene"] == 1 ].iloc[0][:]
    print(obscene_row)
    print(obscene_row["comment_text"])
    
    #get its index
    index = np.where(train["obscene"] == 1)[0][0]
    
        #so we are getting the index of the first sample that is obscene
    print("index:",index)
    pass

def create_unigrams(corpus, train_len, save_pickle = True):
    start_time = time.time()
    
    tfidf = TfidfVectorizer(min_df = 10, strip_accents = "unicode", analyzer = "word", ngram_range = (1,1), use_idf = True, smooth_idf = True, sublinear_tf = True, stop_words = "english")
                    #max_features = 30000 could work well
                    
    tfidf.fit(corpus)
    
    train_unigrams = tfidf.transform(corpus.iloc[:train_len])
    test_unigrams = tfidf.transform(corpus.iloc[train_len:])

    #train_unigrams = tfidf.transform(corpus)
    
    end_time = time.time()
    print("total time with tfidf fit and transform:", end_time - start_time)
    
    if save_pickle:
        pickle_folder = "Pickles"
        os.makedirs(pickle_folder, exist_ok = True)
        
        pickle.dump(tfidf, open(pickle_folder + "/trained_tfidf.pk","wb"))
        pickle.dump(train_unigrams, open(pickle_folder + "/train_unigrams.pk", "wb") )
        pickle.dump(test_unigrams, open(pickle_folder + "/test_unigrams.pk", "wb") )
        print("pickles saved.")
        
    #return train_unigrams, test_unigrams
    
    pass

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]   #argsort returns the indices that would sort an array
    print("topn_ids:", topn_ids)
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def load_vectorizer():
    #we already trained the tfidf in create_unigrams. Just load those pickles.
    pickle_folder = "Pickles"
    
    tfidf = pickle.load(open(pickle_folder + "/trained_tfidf.pk","rb"))
    train_unigrams = pickle.load(open(pickle_folder + "/train_unigrams.pk", "rb") )
    test_unigrams = pickle.load(open(pickle_folder + "/test_unigrams.pk", "rb") )
    print("pickles loaded.")
    
    return tfidf, train_unigrams, test_unigrams

def play_with_tfidf():
    top_n = 25
    row_id = 6 #first obscene row
    
    tfidf, train_features, test_features = load_vectorizer()
    
    vocabulary_size = len(tfidf.vocabulary_)
    print("vocabulary size:", vocabulary_size)
    
    print("top ten tfdif features:")
    print(sorted(tfidf.vocabulary_.items(), key = operator.itemgetter(1), reverse = True)[:top_n])
    print("first ten tfdif features:")
    print(tfidf.get_feature_names()[:top_n])
    
    print("Lets have a look at the first train element.")
    print("type:",type(train_features[row_id]))
    
    #test and train features are sparse matrices. So a record needs to be "unrolled" first 
        #before its features can be accessed
    row = np.squeeze(train_features[row_id].toarray())
    print("row[row_id]:",row)
   
    print("top ten tfidf values in this sample:")
    print(sorted(row)[::-1][:top_n])
            #this just prints the top tfidf values in this record, not the features corresponding to the values
   
    print("top ten tfidf features in this sample:")
    print(top_tfidf_feats(row, tfidf.get_feature_names(),top_n))
            #now we see the features along with their tfidf values
    
    pass

def old_tfidf_creation():
    print("reading data")
    train, test, merged = read_data()
    print("data reading done")
    
    print("\n\ncreating clean_corpus.")
    
    clean_corpus = merged["clean_text"].apply(lambda x: str(x))
    
    print(clean_corpus[0])
    print("type:",type(clean_corpus[0]))
    print("shape:",clean_corpus.shape)

    #find the rows where clean_text == "emptytext"
    empty_rows = merged[ merged["clean_text"] == "emptytext"]
    print("empty rows shape:", empty_rows.shape)
    print("first:")
    print(empty_rows.iloc[0][["comment_text","clean_text"]])
    print("last:")
    print(empty_rows.iloc[ empty_rows.shape[0] -1 ][["comment_text", "clean_text"]])
    
    
    #was having an issue to run tfidf on clean_text. 
    #Was crashing after encountering an np.NaN value. 
        #Not sure where that np.NaN was coming from.
    
    #comment_text itself does not cause any problem in tfidf.
    #so the problem is with the clean_text only.
    #Check the cleaning function again to figure out why this is happening. 
    
    #Update1: tfidf seems working fine when values are converted to string.
    #But shows np.NaN error without it (I still don't undertand why it would though.)
    #Problem with this approach: it shows memory error in the VM while converting to string if > 30000 rows.
    #One approach seems to be to chop off the data, convert, and concat.
    #Try some of those in Tesla/Herc to see if the problem goes away.
    
    #UPDATE2:
    #    Solved by applying the string conversion lambda. But why? :-s
    
    
    #print("checking for empty values")
    
    empty_values = merged[merged["clean_text"].apply(lambda x: len(str(x)) <= 1)]
    print("empty_values shape:", empty_values.shape)
    
    
    print("starting tfidf training ...")
    create_unigrams(clean_corpus, train.shape[0], save_pickle=True)
    print("tfidf done.")
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    According to FeatureUnion example, this should inherit BaseEstimator
    and TransformerMixin. Will need to read them up to find out why.
    '''
    
    #making an extremely simplistic wrapper class. 
    #may improve it later if needed.
    
    #this would only work with one column at a time. So need to call it
    #once for every feature we are using.
    
    #TODO: check sklearn.feature_extraction.Dictvectorize 
    #if we want to use multiple features at once.
    
    
    def __init__(self, feature):
        self.feature = feature
    
    def fit(self, X, y = None):
        #nothing to fit. Literally do nothing.
        #print("inside FeatureSelector.fit()")
        return self
    
    def transform(self, X):
        #nothing to transform. Just return the selected columns.
        res = X[self.feature]
        #        print("inside FeatureSelector.transform()")
        #        print("feature:", self.feature)
        #        print("return shape:", res.shape)
        return res
        
    pass

class ArrayCaster(BaseEstimator, TransformerMixin):
    '''
    This class was written after we initially got an error while using 
    Tfidf and other features together inside a FeatureUnion:
        "ValueError: blocks[0,:] has incompatible row dimensions"
        
    So clearly, some matrix was not in correct shape.
    
    Now, all this class do is to convert a pandas object to a matrix,
    and transpose it. This may actually cause unintended consequences in some
    cases if we pass a 2D dataframe to it. The retured matrix will be the 
    transpose of the given object.
    
    A = pd.DataFrame([ [1,2,3],[4,5,6]])

    A.shape
    Out[75]: (2, 3)
    
    B = np.matrix(A)
    
    B
    Out[77]: 
    matrix([[1, 2, 3],
            [4, 5, 6]])
    
    np.transpose(A)
    Out[81]: 
       0  1
    0  1  4
    1  2  5
    2  3  6
    
    np.transpose(B)
    Out[82]: 
    matrix([[1, 4],
            [2, 5],
            [3, 6]])
    
    
    But for a pandas series, the behaviour is different!
    
    A
    Out[83]: 
       0  1  2
    0  1  2  3
    1  4  5  6
    
    C = A[1]
    
    C
    Out[85]: 
    0    2
    1    5
    Name: 1, dtype: int64
    
    D = np.matrix(C)

    D
    Out[89]: matrix([[2, 5]])
    
    np.transpose(D)
    Out[90]: 
    matrix([[2],
            [5]])
    
    So, converting a pandas Series object to a np matrix converts it to a row
    vector from a column vector. I guess this is the reason the shape mismatch 
    occurs.
    
    By transposing it, we are bringing it back to the column vector.
    
    TLDR:
        Converting a pandas Series to a numpy matrix makes it a row vector from
        column vector. This causes shape mismatch in further operations.
        So instead of implicit conversions inside the functions,
        we are doing it here manually and then transposing the resultant marix,
        to bring it back to the original column vector shape.
        
        This funcion will only work for a single pandas column.
    
    #TODO: Apparently FeatureUnion demands that each constituent transformer
        returns a 2D array. So returning a Series was causing a problem.
        Dig it further.
    '''
    def fit(self, X, y= None):
        return self
        
    def transform(self, data):
        return np.transpose(np.matrix(data))
        
    
    
def make_bare_bone_model():
    '''
    As the name suggests. Just make a model using word and char based tfidf,
    along with the engineered features. Fit, predict and judge.
    Don't worry abut pipelining just yet.
    '''
    
    print("reading data")
    train, test, merged = read_data()
    print("data reading done")
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    #just try with one first
    target_class = "toxic"
    
    print("working with target class:", target_class)
    
    '''
    At first, forget about feature union. 
    Just create the vectorizers, fit them. Transform.
    Concat the transformed rows. (can we?)
    Concat the engineered features.
    Train.
    '''
    
    '''
    Update: It does look like we will need FeatureUnion to combine the sparse
    matrices along with the other columns. There is a sklearn tutrial about it.
    Check.
    '''
    
    '''
    Update 2: 
        For the train data X, we need to perfrom tfidf on clean_text column.
        At the same time, we need some other columns for Logistic Regression.
        
        So we need a more complex setup with both FeatureUnion and Pipelines.
        
        We need an ItemSelector class that just slices one or more columns from X.
        Then pipeline that with the tfidfvectorizers to slice only the clean_text column.
        
        Then use FeatureUnion to glue them together.
        
        Finally, do another Pipeline to add ligistic regression with it.
    '''
    
    #columns we have:
        #        ['id', 'comment_text', 'count_sent', 'count_word', 'count_unique_word',
        #       'count_letters', 'count_punctuations', 'count_words_upper',
        #       'count_words_title', 'count_stopwords', 'mean_word_len',
        #       'word_unique_percent', 'punct_percent', 'spam', 'toxic', 'severe_toxic',
        #       'obscene', 'threat', 'insult', 'identity_hate', 'clean', 'clean_text']
    
    #make word-vectorizer
    word_vectorizer = TfidfVectorizer(sublinear_tf = True, 
                                      strip_accents = "unicode", 
                                      analyzer = "word",
                                      min_df = 10,
                                      max_features = 30000,
                                      ngram_range = (1,1),
                                      use_idf = True,
                                      smooth_idf = True
                                      )
       
    #make char-vecorizer
    char_vectorizer = TfidfVectorizer(sublinear_tf = True,
                                      strip_accents = "unicode",
                                      analyzer = "char",
                                      min_df = 10,
                                      max_features = 30000,
                                      ngram_range = (1,4),
                                      use_idf = True,
                                      smooth_idf = True
                                      )
    
#    engineered_features = [ 'count_sent', 'count_word', 'count_unique_word',
#                           'count_letters', 'count_punctuations', 'count_words_upper',
#                           'count_words_title', 'count_stopwords', 'mean_word_len',
#                           'word_unique_percent', 'punct_percent', 'spam']
#    
    engineered_features = ["spam", "count_stopwords"]
    
    engineered_features_transformers = [ (feature_name, Pipeline([
                            ("selector", FeatureSelector(feature = feature_name)),
                            ("caster", ArrayCaster())
                            ])) for feature_name in engineered_features]
    
    transformer_list = [
                    #word vectorizer        
                    ("word_vect", Pipeline([
                            ("selector", FeatureSelector(feature = "clean_text")),
                            ("tfidf", word_vectorizer)
                            ])),
                    
                    #char vectorizer
                    ("char_vect", Pipeline([
                            ("selector", FeatureSelector(feature = "clean_text")),
                            ("tfidf", char_vectorizer)
                            ]))
                    ]
    
    #transformer_list = []
    transformer_list.extend(engineered_features_transformers)
    
    #join engineered features
    feat_union = FeatureUnion( 
                transformer_list = transformer_list,
                n_jobs = 1,
                #transformer_weights = []
            ) 
        
    classifier = LogisticRegression(solver = "sag", n_jobs = 1, C = 1.0)
    
    pipeline = Pipeline( [ ("union", feat_union), ("clf", classifier) ] )
    
    #TODO: the way things are, we are calling tfidf directly.
    #So we can only train it over the train data we have, not on train and test together.
    #To overcome that, make a wrapper around it.

    #get X and y
    no_of_rows = 10000
    
    train_X = pd.concat([train.iloc[0:no_of_rows, 0:14],train.iloc[0:no_of_rows]["clean_text"]], axis = 1)
    
    print(train_X.shape)
    print(type(train_X))
    print(train_X.columns)

    train_y = train.iloc[0:no_of_rows][target_class]
    print(train_y.shape)
    print(type(train_y))
    
    
    #predict and test
    start_time = time.time()
        
    cv_score = np.mean(cross_val_score(pipeline, train_X, train_y, cv = 3, scoring = "roc_auc", n_jobs = 3))
    
    end_time = time.time()
    print("CV time:", end_time - start_time)
    
    #scores.append(cv_score)
    print("cv_score for class {} : {}".format(target_class, cv_score))
    
    pass

def do_grid_search():
    '''
    A very small grid search for all the features. Just to make sure that we know how to do it
    with a complicated collection of pipelines and feature_unions.
    '''
    pass


def make_feature_unions():
    '''
    We are re-using this code a lot.
    So pack it in its own function.
    '''
    
    #make the feature_unions one by one
    word_vectorizer = TfidfVectorizer(sublinear_tf = True, 
                                      strip_accents = "unicode", 
                                      analyzer = "word",
                                      min_df = 10,
                                      max_features = 30000,
                                      ngram_range = (1,1),
                                      use_idf = True,
                                      smooth_idf = True
                                      )
       
    char_vectorizer = TfidfVectorizer(sublinear_tf = True,
                                      strip_accents = "unicode",
                                      analyzer = "char",
                                      min_df = 10,
                                      max_features = 30000,
                                      ngram_range = (1,4),
                                      use_idf = True,
                                      smooth_idf = True
                                      )
    
    # a small list of engineering features
    engineered_features = ["spam"]
    
    transformer_list = [
                        #word vectorizer
                        ( "word_vect", Pipeline([ 
                                ("selector", FeatureSelector(feature = "clean_text")) , 
                                ("tfidf", word_vectorizer) ]) ) , 
                        #char vectorizer
                        ("char_vect", Pipeline([ 
                                ("selector", FeatureSelector(feature = "clean_text")), 
                                ("tfidf",char_vectorizer)]) ) 
                        ]
    
    engineered_features_transformers = [ (feature_name, Pipeline( [ 
                                                        ("selector",FeatureSelector(feature = feature_name)), 
                                                        ("caster", ArrayCaster()) ] )) 
                                        for feature_name in engineered_features]

    #then concat them together
    transformer_list.extend(engineered_features_transformers)
    
    
    #make the actual feature union from this list
    feat_union = FeatureUnion( 
                transformer_list = transformer_list,
                n_jobs = 3,
                #transformer_weights = []
            ) 
    
    return feat_union

def final_model():
    '''
    Practically the same as the bare-bone model. Just for all the features.
    Intention is to make a clean one without the comments and test codes.
    So for documentation, check that one.
    '''
    
    #read data
    print("reading data")
    train, test, merged = read_data()
    print("data reading done")
    
    #train data properties
    print("train data properties:")
    print("shape:", train.shape)
    print("type:", type(train))
    print("columns: ", train.columns)
    
    #test data properties
    print("test data properties:")
    print("shape:", test.shape)
    print("type:", type(test))
    print("columns: ", test.columns)
    
    #merged data properties
    print("merged data properties:")
    print("shape:", merged.shape)
    print("type:", type(merged))
    print("columns: ", merged.columns)
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    #start with one
    #target_class = "toxic"
    
    '''
    Issue: 
        This re-dos the fit_transforms for every iteration of the loop.
        This is unnecessary.
        Only the classifier needs to be re-trained. 
        There sure is a more efficient way.
        
        Though, this is an issue here because we are using the same set of 
            hyper parameters for tfidfs for each feature.
        If we did a grid search and did use optimized parameters for each one,
            this would've been the most efficient way.
    '''
    
    scores = []
    
    no_of_rows = 1000
    
    submission = pd.DataFrame.from_dict( {"id": test.iloc[0:no_of_rows]["id"]} )
    print("submisison data properties (initially):")
    print("shape:", submission.shape)
    print("type:", type(submission))
    print("columns: ", submission.columns)
    print("first 5 samples:")
    print(submission.head(5))
    
    for target_class in class_names:
        
        feat_union = make_feature_unions()
        
        #add the classifier in front
        classifier = LogisticRegression(solver = "sag", n_jobs = 1, C = 1.0)
        
        pipeline = Pipeline([ ("union", feat_union), ("clf", classifier) ])
        
        #get the train and test data ready
        
        train_X = pd.concat([train.iloc[0:no_of_rows, 0:14],train.iloc[0:no_of_rows]["clean_text"]], axis = 1)
        train_y = train.iloc[0:no_of_rows][target_class]
        
        #do cross_validation and get the scores
        start_time = time.time()
            
        cv_score = np.mean(cross_val_score(pipeline, train_X, train_y, cv = 3, scoring = "roc_auc", n_jobs = 1))
        
        end_time = time.time()
        print("CV time:", end_time - start_time)
        
        scores.append(cv_score)
        print("cv_score for class {} : {}".format(target_class, cv_score))
        
        #finally predict
    
        start_time = time.time() 
        pipeline.fit(train_X, train_y)
        end_time = time.time()
        print("fitting time: ", end_time - start_time)
        
        start_time = time.time()
        
        result = pipeline.predict_proba( test.iloc[0:no_of_rows][:] )
                                        #predicting on only a few rows for test purpose
        
        #        print("result data properties:")
        #        print("shape:", result.shape)
        #        print("type:", type(result))
        #        #print("columns: ", result.columns)
        #        print("first three:")
        #        print(result[:3])
        
        submission[target_class] = result[:, 1]
        end_time = time.time()
        print("Prediction time: ", end_time - start_time)
        
    print("submisison data properties (at the end):")
    print("shape:", submission.shape)
    print("type:", type(submission))
    print("columns: ", submission.columns)
    print("first 5 samples:")
    print(submission.head(5))
    
    pass
    
def more_efficient_model():
    '''
    Issues with final_model(): 
        It runs a loop for each category, and 
        ii re-dos the fit_transforms for every iteration of the loop.
        This is unnecessary.
        Only the classifier needs to be re-trained. 
        There sure is a more efficient way.
        
        [Though, this is an issue here because we are using the same set of 
            hyper parameters for tfidfs for each feature.
        If we did a grid search and did use optimized parameters for each one,
            this would've been the most efficient way.]
    '''
    
    #read data
    print("reading data")
    train, test, merged = read_data()
    print("data reading done")
    
    #train data properties
    print("train data properties:")
    print("shape:", train.shape)
    print("type:", type(train))
    print("columns: ", train.columns)
    
    #test data properties
    print("test data properties:")
    print("shape:", test.shape)
    print("type:", type(test))
    print("columns: ", test.columns)
    
    #merged data properties
    print("merged data properties:")
    print("shape:", merged.shape)
    print("type:", type(merged))
    print("columns: ", merged.columns)
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    scores = []
    
    no_of_rows = 1000
    
    submission = pd.DataFrame.from_dict( {"id": test.iloc[0:no_of_rows]["id"]} )
    print("submisison data properties (initially):")
    print("shape:", submission.shape)
    print("type:", type(submission))
    print("columns: ", submission.columns)
    print("first 5 samples:")
    print(submission.head(5))
    
    feat_union = make_feature_unions()
    
    train_X = pd.concat([train.iloc[0:no_of_rows, 0:14],train.iloc[0:no_of_rows]["clean_text"]], axis = 1)
    
    print("Calling fit_transform on X_train ...")
    start_time = time.time() 
    X_fit_transformed = feat_union.fit_transform(train_X)
    end_time = time.time()
    print("fitting time: ", end_time - start_time)

    print("transformed train data properties:")
    print("shape:", X_fit_transformed.shape)
    print("type:", type(X_fit_transformed))
    
    #also, trasform test_X as we are not using pipeline any more
    
    test_X_transformed = feat_union.transform(test.iloc[0:no_of_rows][:])
    print("transformed test data properties:")
    print("shape:", test_X_transformed.shape)
    print("type:", type(test_X_transformed))
    
    
#    submission = pd.DataFrame.from_dict( {"id": test.iloc[0:no_of_rows]["id"]} )
#    print("submisison data properties (initially):")
#    print("shape:", submission.shape)
#    print("type:", type(submission))
#    print("columns: ", submission.columns)
#    print("first 5 samples:")
#    print(submission.head(5))
#    
#        
    for target_class in class_names:
    
        train_y = train.iloc[0:no_of_rows][target_class]
            
        classifier = LogisticRegression(solver = "sag", n_jobs = 1, C = 1.0)
    
        cv_score = np.mean(cross_val_score(classifier, X_fit_transformed, train_y, cv = 3, scoring = "roc_auc", n_jobs = 1))
        
        end_time = time.time()
        print("CV time:", end_time - start_time)
        
        scores.append(cv_score)
        print("cv_score for class {} : {}".format(target_class, cv_score))
        
        start_time = time.time() 
        classifier.fit(X_fit_transformed, train_y)
        end_time = time.time()
        print("fitting time: ", end_time - start_time)
        
        start_time = time.time()
        result = classifier.predict_proba( test_X_transformed )
                                        #predicting on only a few rows for test purpose
        
        #        print("result data properties:")
        #        print("shape:", result.shape)
        #        print("type:", type(result))
        #        #print("columns: ", result.columns)
        #        print("first three:")
        #        print(result[:3])
        
        submission[target_class] = result[:, 1]
        end_time = time.time()
        print("Prediction time: ", end_time - start_time)
        
    print("submission data properties (at the end):")
    print("shape:", submission.shape)
    print("type:", type(submission))
    print("columns: ", submission.columns)
    print("first 5 samples:")
    print(submission.head(5))
    
        
    pass


if __name__ == "__main__":
    print("Hello World")
    
    #play_with_tfidf()
    #read_data_and_query()
    #make_bare_bone_model()
    #final_model()
    more_efficient_model()
    
    print("The world never says hello back :-(")