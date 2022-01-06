#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:02:29 2022

@author: oleksandryatsenko
"""
# import packages

# =============================================================================
# import packages
# =============================================================================

import sys

# data extraction
from sqlalchemy import create_engine

# standard data processing
import pandas as pd
import numpy as np
import sys
import time

# tokenization
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# visualization libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn liabraries

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# saving model results
import joblib

# =============================================================================
# Load data
# =============================================================================

def load_data(database_filepath):
    '''
    INPUT:
    none

    OUTPUT:
    X,y - extracting features and labels from TweetsDatabase
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('TweetsDatabase', engine)

    # defining features - X and labels y
    # dropping 'child alone feature' as this is absent in our data
    X = df['message']
    y = df.drop(['id','original','message','genre','child_alone'], axis = 1)

    # Replacing rougue 2 values, otherwise this will cause an error in confusion matrix
    y.related.replace('2','1',inplace=True)
    y = y.astype(int)
    return X,y


# =============================================================================
# Tokenization function to process our text data
# =============================================================================

def tokenize(text):
    '''
    INPUT:
    text - individual tweets from TweetsDatabase in string format

    OUTPUT:
    clean_tokens - list of cleaned tokens

    '''
    # text cleaning
    text = text.lower() # Convert to lowercase
    text = re.sub('[^A-Za-z0-9]+',' ', text) # removing punctuation
    words = nltk.word_tokenize(text) # tokenizing words
    stop_words = set(stopwords.words('english')) # adding set of stopwords
    filtered_words = [w for w in words if not w.lower() in stop_words] # filtering words

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in filtered_words:
        # lemmatization
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    return clean_tokens

# =============================================================================
# Assembling model evaluation components
# =============================================================================

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    '''
    INPUT:
    confusion_matrix - results of sklearn confusion_matrix
    axes
    class_label
    class_names
    fontsize

    OUTPUT:
    confusion matrix chart

    '''
    # plotting heatmap of confusion matricies
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)

def model_results(y_test, y_pred):
    '''
    INPUT:
    y_test
    y_pred


    OUTPUT:
    labels - feature names (columns)
    ax
    fig
    vis_arr - multilabel confusion matrix results
    accuracy - model's accuracy
    '''
    # extracting list of labels from column names
    labels = list(y_test.columns.values)

    # we use multilabel confusion matrix since we have multiple labels
    vis_arr = multilabel_confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(4, 4, figsize=(18, 9))
    accuracy = (y_pred == y_test).mean()

    return labels, ax, fig, vis_arr, accuracy

# =============================================================================
# Build a machine learning pipeline
# =============================================================================

def model_pipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])

    # HINT: to check the full set of all parameters we can use in gridsearch - pipeline.get_params()
    parameters = {
        'tfidf__use_idf': (True, False)

        # 'clf__n_estimators': [50, 100, 200],
        # 'clf__min_samples_split': [2, 3, 4],
        # 'tfidf__use_idf': (True, False),
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3)
    return cv

# =============================================================================
# Train model
# =============================================================================

def build_model(X_train, y_train):
    '''
    INPUT:
    X_train
    y_train

    OUTPUT:
    model
    '''
    model = model_pipeline()
    model.fit(X_train, y_train)

    return model

# =============================================================================
# Evaluate your model
# =============================================================================

def model_evaluation(model, X_train, X_test, y_train, y_test):
    '''
    INPUT:
    model - sklearn model
    X_train
    X_test
    y_train
    y_test

    OUTPUT:
    charts
    '''

    # predicting results for X_test
    y_pred = model.predict(X_test)

    labels, ax, fig, vis_arr, accuracy = model_results(y_test, y_pred)

    for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    # displaying results
    fig.tight_layout()
    plt.show()
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)

# =============================================================================
# Export your model as a pickle file
# =============================================================================

def save_model(model, model_filepath):
    '''
    INPUT:
    model

    OUTPUT:
    pickle file of a model
    '''
    joblib.dump(model, open(model_filepath, "wb"))

# =============================================================================
#  Main function
# =============================================================================

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        start_time = time.time()
        print('Loading data..')

        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        print('Training model...')
        model = build_model(X_train, y_train)

        print('Saving model...')
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Evaluating model..')
        print('Close Confusion matrix results to exit')
        model_evaluation(model, X_train, X_test, y_train, y_test)

        # predictions running time
        print("Running time:","--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
