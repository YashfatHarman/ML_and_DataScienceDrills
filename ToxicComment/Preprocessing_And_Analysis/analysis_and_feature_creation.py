#create derived (and may be leaky) features
#save

#train models
#win

print("Hello World!")

#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time

start_time  = time.time()
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

end_time = time.time() 

print("all module imported peacefully.")

print("Module loading time:", end_time - start_time)

start_time = time.time()

#import dataset
train = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")

end_time = time.time()
print("data reading time:", end_time - start_time)

print("train shape:", train.shape)
print("test shape:", test.shape)

print("train columns:", train.columns)
print("test columns:", test.columns)

#print("train first five:")
#print(train.head(5))
#print("test first five:")
#print(test.head(5))

#crete custom columns

#first create a clean column
rowsum = train.iloc[:, 2:].sum(axis=1)
train["clean"] = (rowsum == 0).astype(int)

#count types of each type of comments
sum = train.iloc[:,2:].sum()
#print(sum)

#lets make a plot
ax = sns.barplot(sum.index, sum.values, alpha = 0.8)
plt.title("comment per class")
plt.xlabel("comment class")
plt.ylabel("comment count")

rects = ax.patches
labels = sum.values

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = "center", va = "bottom")

plt.show()



#lets check how many comments have multiple tags
mult_tag_count = train.iloc[:, 2:-1].sum(axis=1)
val_freq = mult_tag_count.value_counts()

#make a plot again
ax = sns.barplot(val_freq.index, val_freq.values, alpha = 0.8)
plt.title("multiple tags per comment")
plt.xlabel("# of tags")
plt.ylabel("# of occurances")

rects = ax.patches
labels = val_freq.values

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = "center", va = "bottom")
           
plt.show()



# lets start feature engineering

# direct features:
    # Word frequency:
        # Count features
        # Bigrams
        # Trigrams    
    # Vector distance mapping of words
    # Sentiment Scores

# Indirect features:
    # Count of sentences
    # Count of words
    # Count of unique words
    # Count of letters
    # Count of punctuations
    # Count of uppercase words/ letters
    # Count of stop words
    # avg length of each word
    
# Leaky features:
    # tocix IP scores
    # toxic users

# In this case, there may not be a lot of 
#    leaky features present in the test set.
#    So we may just ignore these.
# Also, leaky features will certainly lead to overfitting (might not be a bad thing
# in this case though. Our test set is fixed.)

#merge test and train before extracting indirect features.

print("starting creation of custom features.")
start_time = time.time()

merge = pd.concat([train.iloc[:,0:2], test.iloc[:, 0:2]])
print(merge.shape)
#print(merge.head(5))
#print(merge.tail(5))

##not sure what difference it makes
#Answer:
#    reset_index, as the name suggests, resets index.
#    Each of train and test set entries have an index, starting from 0 to somewhere around 153K.
#    When they are merged, the indexes do not change, which means now we have two rows with
#    index 0, two with index 1, etc. 
#    reset_index resets the index in the merged set so now the indexes are in the 
#    range 0 to 312K.
#    Checking the output of tail(5) in the reset and non-reset set will make it clear.
merged = merge.reset_index(drop = True)
print(merged.shape)
#print(merged.head(5))
#print(merged.tail(5))

#lets make the indirect features.
# count_sent, count_word,count_unique_word, count_letters
# count_punctuations, count_words_upper, count_words_title
# count_stopwords, mean_word_len
# word_unique_percent, punct_percent

merged["count_sent"] = merged["comment_text"].apply(lambda x: len(re.findall("\n", str(x))) + 1)

merged["count_word"] = merged["comment_text"].apply(lambda x: len(str(x).strip().split()))

merged["count_unique_word"] = merged["comment_text"].apply(lambda x : len(set(str(x).split())))

merged["count_letters"] = merged["comment_text"].apply(lambda x : len(str(x)))

merged["count_punctuations"] = merged["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation ]))

merged["count_words_upper"] = merged["comment_text"].apply(lambda x : len([w for w in str(x).split() if w.isupper()]))

merged["count_words_title"] = merged["comment_text"].apply(lambda x : len([w for w in str(x).split() if w.istitle()]))

eng_stopwords = set(stopwords.words("english"))

merged["count_stopwords"] = merged["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

merged["mean_word_len"] = merged["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

merged["word_unique_percent"] = merged["count_unique_word"]*100/merged["count_word"]

merged["punct_percent"] = merged["count_punctuations"]*100/merged["count_word"]

merged["spam"] = (merged["word_unique_percent"] < 0.30).astype(int)

end_time = time.time()

print(merged.shape)
print(merged.columns)
#print(merged.head(5))

print("custom feature creation time: ", end_time - start_time)

#now separate train and test again
train_feats = merged.iloc[0:len(train)]
test_feats = merged.iloc[len(train):]
print("train_feats shape:", train_feats.shape)
print("test_feats shape:", test_feats.shape)

#join the tags with train_feats
train_tags = train.iloc[:, 2:]
train_feats = pd.concat([train_feats, train_tags], axis = 1)
print("train_feats shape:", train_feats.shape)
#print(train_feats.head(5))

#print("an example toxic spam:")
#print(train_feats[ (train_feats["spam"] == True) & (train_feats["clean"]==False) ].iloc[0]["comment_text"])
#
#print("an example clean spam:")
#print(train_feats[ (train_feats["spam"] == True) & (train_feats["clean"]==True) ].iloc[0]["comment_text"])

###save it as csv file
train_feats.to_csv("../Data/train_feats.csv", index = False)
test_feats.to_csv("../Data/test_feats.csv", index = False)

print("The world never says hello back.")





