# Disaster Response Pipeline Project

## Table of Contents
1. [Installation](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#installations)
2. [Project Motivation](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#project-motivation)
3. [Folder & Files Descriptions](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#folder--files-descriptions)
4. [Instructions](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#instructions)
5. [Results](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#results)
6. [Licensing, Authors, and Acknowledgements](https://github.com/Olek-ua/Disaster-Response-Pipelines/tree/test#Licensing,-Authors,-and-Acknowledgement)

## Installations
You would need to install some of the standard libraries:

for data analysis & viz:
- numpy
- pandas
- sqlalchemy
- matplotlib
- seaborn
- datetime
- time

for text processing:
- re
- nltk
- joblib

and for data modelling we will use only:

- scikit-learn (version 0.23.2 or higher)

The easiest way to install package is using `pip`

`pip install -U <package name>`

or `conda`:

`conda install -c conda-forge <package name>`

make sure to replace `<package name>` with actual package name, like:

`conda install -c conda-forge scikit-learn`

## Project Motivation

In this project we aim to process great amount of messages from Twitter to identify
those related to various disasters. This will help us to filter down to the most relevant ones to prioritise response. A tool like this can be used by
emergency services for monitoring messages online.

Overview of Training Dataset:
![Training Dataset](https://github.com/Olek-ua/Disaster-Response-Pipelines/blob/test/screenshots/Overview%20of%20Training%20Data%20Set.png)

To achieve our target we will pre-process messages by using bag-of-words method and feeding its results into
the model which brings us the highest accuracy and F1 score.

## Folder & Files Descriptions

- **jupyter_files** - this folder contains jupyter workbook with all of the key steps
for ML Pipeline preparation and testing.

- **workspace** - main folder containing all the completed ETL and ML pipelines as well as
database and csv files used for training the model. You would mostly use this folder.

## Instructions:
0. Clone the repository `$ git clone https://github.com/Olek-ua/Disaster-Response-Pipelines`
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results:

Prediction results are vary depending on a feature. Looking into F1 score as
balanced metric between precision and recall we can clearly see that there is a lot space for improvement.

Model statistics:
![Model results](https://github.com/Olek-ua/Disaster-Response-Pipelines/blob/test/screenshots/Attributes%20statistics%20-%20SVC%20Classifier.png)

Confusion matrix:
![Confusion matrix](https://github.com/Olek-ua/Disaster-Response-Pipelines/blob/test/screenshots/Disaster_Response_Confusion_Matrix_Results.png)

The key reasons why results are not completely satisfactory:

- Imbalanced data set
  - labels are very likely not to be equally represented in test and control groups.
  - This issue can be overcome with stratified sampling, however
since we are dealing with multi-label classification problem
this is technically very challenging.
 - We can increase the test size, but this may lead to overfitting. However combined with proper cross validation may bring better results.

## Licensing, Authors, and Acknowledgements:

The content of this repository is licensed under a [Creative Commons Attribution License](https://creativecommons.org/licenses/by/3.0/us/)
