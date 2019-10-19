import pandas as pd
import numpy as np

import time

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




#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

def clean(comment):
    #convert to lower case
    comment = comment.lower()
    
    #remove newline
    comment = re.sub("\\n", " ", comment)
    
    #remove leaky elements like ip address
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    
    #remove usernames
    comment = re.sub("\[\[.*\]", "", comment)
    
    #split the sentences into words
    words = tokenizer.tokenize(comment)
    
    #print(words)
    
    #replace aphostrophe
    words = [APPO[word] if word in APPO else word for word in words]
    #print(words)
    words = [lem.lemmatize(word,'v') for word in words]
    #print(words)
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent = " ".join(words)
    
    #remove any non-alphanumeric
    clean_sent=re.sub("\W+"," ",clean_sent)
    
    clean_sent=re.sub(r"\s+"," ",clean_sent)
    
    #TODO
    #do a check here if it is empty string. Handle the case.
    #Make sure the returned string is a string/unicode.
    #That might take care of the tfidf problem.
    
    return clean_sent

def clean_comment_text():
    #train = pd.read_csv("../Data/train_feats.csv", index_col = [0,1])
    #test = pd.read_csv("../Data/test_feats.csv", index_col = [0,1])
        # was necessary at one point. not anymore.
        
    train = pd.read_csv("../Data/train_feats.csv")
    test = pd.read_csv("../Data/test_feats.csv")
    
    #clean corpus
    
    #merged = pd.concat([train.iloc[:,0:14], test.iloc[:, 0:14]])
    #print(merged.shape)
    #
    #corpus = merged.comment_text
    
    #print(corpus.iloc[12235])
    #print(clean(corpus.iloc[12235]))
    
    train["clean_text"] = train["comment_text"].apply(clean)
    test["clean_text"] = test["comment_text"].apply(clean)
    
    train.to_csv("../Data/train_feats_clean.csv", index = False)
    test.to_csv("../Data/test_feats_clean.csv", index = False)
    
    print("file writing done.")
    
def find_and_replace_empty_clean_text():
    '''
    Turns out, some comment texts got empty after all those cleaning and pruning.
    This poses an issue for tfidfvectorizer.
    So replacing them with a pre-decided static text ("emptytext")
    '''
    train = pd.read_csv("../Data/train_feats_clean.csv")
    test = pd.read_csv("../Data/test_feats_clean.csv")
    
    print("file reading done. Applying transformation.")
    
    train["clean_text"] = train["clean_text"].apply(lambda x: "emptytext" if len(str(x)) <= 1 else str(x))
    test["clean_text"] = test["clean_text"].apply(lambda x: "emptytext" if len(str(x)) <= 1 else str(x))
    
    train.to_csv("../Data/train_feats_clean.csv", index = False)
    test.to_csv("../Data/test_feats_clean.csv", index = False)
    
    print("file writing done.")
    
print("Hello World!")

print("calling clean_comment_text()")

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()
clean_comment_text()

print("end of clean_comment_text()")


print("\ncalling find_and_replace_empty_clean_text")
find_and_replace_empty_clean_text()
print("Done with find_and_replace_empty_clean_text.")

train_new = pd.read_csv("../Data/train_feats_clean.csv")
test_new = pd.read_csv("../Data/test_feats_clean.csv")
print(train_new.shape)
print(train_new.columns)
print(train_new.iloc[0])

#check if the empty_replacing was done properly
empty_train = train_new[train_new["clean_text"] == "emptytext"]
print("empty train:")
print(empty_train.shape)
print(empty_train.iloc[0][["comment_text", "clean_text"]])
print(empty_train.iloc[1][["comment_text", "clean_text"]])
print(empty_train.iloc[2][["comment_text", "clean_text"]])

empty_test = test_new[test_new["clean_text"] == "emptytext"]
print("empty test:")
print(empty_test.shape)
print(empty_test.iloc[0][["comment_text", "clean_text"]])
print(empty_test.iloc[1][["comment_text", "clean_text"]])
print(empty_test.iloc[2][["comment_text", "clean_text"]])

print("\nthe world never says hello back :-(")