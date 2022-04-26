# Disaster Response Pipeline Project

## Instalations:
1. Python 3.7.9  
1. pandas 1.3.5  
2. plotly 5.7.0  
3. nltk 3.7  
4. Flask 2.1.1  
5. scikit-learn 1.0.2  
6. SQLAlchemy 1.4.35  
7. numpy 1.21.6  
8. joblib 1.1.0  

## Project Motivation:
The motivation of this project is to use disaster data to build a model that classifies messages. The project goes through the stages of
a Machine Learning Pipeline with the ETL process, modeling and deploying an app.

## File Descriptions:
The project contains three main files:
1. process_data.py: Makes all transformations in disaster_categories.csv and disaster_messages.csv to load them into the database
DisasterResponse.db.
2. train_classifier.py: Makes predictions and evaluations for all categories in the data.  
3. run.pu: The file that run the visual application to make predictions within new words.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements:
Thanks to udacity for bringing quality content about the pipeline process of a machine learning model
