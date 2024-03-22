import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
# Splitting the data into a training set, validation set and a test set
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

# Initializing TreebankWordTokenizer
## Reference: https://www.nltk.org/_modules/nltk/tokenize/treebank.html

Tokenizer = nltk.TreebankWordTokenizer();

## Here I am using a context window of 10 and get the mean vector of words within the context window.
## In cases where there is an unknown word is beside the unknown word, that unknown word's meaning will
## be found first, thus a recursion is used here.

def gettingVectorsOfUnknownWords(tokens, i, dictOfUnknown):
    
    ##Use dictionary to map unknown words
    
    ## Setting context window, context window of 10 is used
    lower = max(i-10, 0);
    higher = min(i+10, len(tokens)-1);
    current = lower;
    firstValueW2V = w2v[next(iter(w2v))];
    vectorReturn = np.zeros(firstValueW2V.shape);
    count = 0;
    tmpCount=0;
    haveUnknownTerms = False;
    listOfUnknownWord = [];
    while current<=higher:
        if current == i:
            current +=1;
            continue;
        else:
            if (tokens[current].lower() in w2v):
                vectorReturn += w2v[tokens[current].lower()];
                tmpCount += 1;
            else:
                #if (tokens[current].lower() in dictOfUnknown):
                if (current in dictOfUnknown):
                    #vectorReturn += dictOfUnknown[tokens[current].lower()];
                    vectorReturn = np.add(vectorReturn, dictOfUnknown[current]);
                    tmpCount+=1;
                else:
                    haveUnknownTerms = True;
                    listOfUnknownWord.append(current);
            current+=1;
            count += 1;
    temporaryMeaning = np.array(vectorReturn)/tmpCount;
    if (haveUnknownTerms):
        #dictOfUnknown[tokens[i].lower()] = temporaryMeaning;
        dictOfUnknown[i] = temporaryMeaning;
        for j in listOfUnknownWord:
            unknownWordVector, dictOfUnknown = gettingVectorsOfUnknownWords(tokens, j, dictOfUnknown);
            #print(unknownWordVector.shape);
            dictOfUnknown[j] = unknownWordVector;
    for k in listOfUnknownWord:
        vectorReturn += dictOfUnknown[k];
    vectorReturn = vectorReturn/count;
    #vectorList[i] = vectorReturn;
    #if (haveUnknownTerms):
        
    return vectorReturn, dictOfUnknown;

# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # TODO: tokenize the input document
    
    tokens = Tokenizer.tokenize(doc);
    
    #Remove stopwords and punctuations
    
    # TODO: aggregate the vectors of words in the input document
    
    # Mean of all the vectors of words in the input document is
    # used to represent the vector of the input document.
    
    vectorsList = []
    
    dictOfUnknown = {};

    i=0;
    while i < len(tokens):
        if tokens[i].lower() in w2v:
            vectorsList.append(w2v[tokens[i].lower()]);
        else:
            vectorToAdd, dictOfUnknown = gettingVectorsOfUnknownWords(tokens, i, dictOfUnknown);
            vectorsList.append(vectorToAdd);
        i += 1;
    
    count = 1;
    vec = vectorsList[0];
    while count<len(vectorsList):
        vec = np.add(vec, vectorsList[count]);
        count+=1;
    
    vec = vec/len(vectorsList);


    return vec;



# fit a linear model
## Reference: COMP6490 Lab 3 Practical Exercise

def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    #TODO: convert each of the training documents into a vector
    docList = [];
    for i in Xtr:
        docVector = document_to_vector(i);
        docList.append(docVector);
        
    model = LogisticRegression(C=C, max_iter=10000).fit(docList, Ytr);

    #TODO: train the logistic regression classifier
    return model



# fit a linear model 
## Reference: COMP6490 Lab 3 Practical Exercise

def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    #TODO: convert each of the testing documents into a vector
    testDocList = [];
    for i in Xtst:
        docVector = document_to_vector(i);
        testDocList.append(docVector);
    #TODO: test the logistic regression classifier and calculate the accuracy
    testResult = model.predict(testDocList);
    #print(testResult);
    score = accuracy_score(Ytst, testResult);
    #score=0;
    return score

# TODO: search for the best C parameter using the validation set

resultRecord = {};
c = 1;
rMax = 0;
cMax = 0;
i = 8;
while i>=0.25:
    patience = 0;
    while (patience < 3):
        if (c in resultRecord):
            tmpR = resultRecord[c];
        else:
            model = fit_model(X_train, Y_train, c);
            tmpR = test_model(model, X_val, Y_val);
            resultRecord[c] = tmpR;
        print(str(tmpR) + "   and    c: " + str(c));
        if (tmpR >= rMax):
          cMax = c;
          rMax = tmpR;
          patience = 0;
        else:
          patience += 1;
        c += i;
    c = max(0,cMax-i);
    i = i/2;
print(str(rMax) + "for the chosen C: " + str(cMax));

# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result
newTrainingSet = Xr[0:val_end];
newTrainingAnswer = Yr[0:val_end];
modelFinal = fit_model(newTrainingSet, newTrainingAnswer, cMax);
print(test_model(modelFinal, X_test, Y_test));