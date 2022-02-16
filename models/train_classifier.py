import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """
    Function to load the data from the database
    Input:
        database_filepah - The database filepath to make predictions
    Output:
        X - Columns with the values com make the predict 
        Y - Columns with the variable responses
        category_names - Name of the columns with the variable responses
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'original', 'message','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize the text
    Input:
        text - Text from message
    Output:
        toknes - Tokens from the text
    """
    # normalise case and remove pontuation
    text = re.sub(r'[^A-Za-z0-9]', ' ', text.lower())
    
    # tokenization
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    Function to apply the model pipeline and apply GridSearchCV
    Output:
        best - Model with the best parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False)
    }
    
    best = GridSearchCV(pipeline, parameters, verbose=1)
    
    return best


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to show the reports of the model to each category
    Input:
        model - Model object
        X_test - X Data to test
        Y_test - Y data to test
        category_names - Name of the response variables
    """
    predicts = train_model.predict(X_test)
    dict_results = {}
    for i, col in enumerate(Y_test):
        print('Column:', col)
        print(classification_report(Y_test[col], predicts[:, i]))


def save_model(model, model_filepath):
    """
    Function to save the model in pickle
    Input:
        model - model object
        model_filepath - destination path of the pickle
    """
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()