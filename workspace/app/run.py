import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

import os
import re

print(re.sub('\/app$', '', os.path.abspath('.')))


# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')

# works
# engine = create_engine('sqlite:////' + re.sub('\/app$', '', os.path.abspath('.')) + '/data/DisasterResponse.db')

df = pd.read_sql_table('TweetsDatabase', engine)


genre_counts = df.groupby('genre').count()['message']

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    d = {'Count':df.drop(['id','message','original','genre'], axis=1).astype(int).sum()}
    top_5_classes = pd.DataFrame(data = d).sort_values(by=['Count'], ascending=False).head(5)['Count']
    top_5_classes_names  = list(top_5_classes.index)

    all_classes = pd.DataFrame(data = d).sort_values(by=['Count'], ascending=False)['Count']
    all_classes_names  = list(all_classes.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_5_classes_names,
                    y=top_5_classes
                )
            ],

            'layout': {
                'title': 'Top 5 Messages Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classes"
                }
            }
        },
        {
          "data": [
            {
              "uid": "f4de1f",
              "hole": 0.1,
              "name": "Class",
              "pull": 0,
              "type": "pie",
              "domain": {
                "x": [
                  0,
                  1
                ],
                "y": [
                  0,
                  1
                ]
              },
              "marker": {
                "colors": [
                  "#7fc97f",
                  "#beaed4",
                  "#fdc086",
                  "#ffff99",
                  "#386cb0"
                ]
              },
              "textinfo": "percent",
              "hoverinfo": "label+text+value",
              "labels": all_classes_names,
              "values": all_classes,
            }
          ],
          "layout": {
            "title": "Classes Shares",
            "width": 1000,
            "height": 800,
          },
          "frames": []
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
