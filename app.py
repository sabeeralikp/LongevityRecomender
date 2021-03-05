# app.py
from flask import Flask, request, Response
app = Flask(__name__)

import pandas as pd
import numpy as np

dataset = pd.read_csv("./data.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
import json


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
dataset['Summary'] = dataset['Summary'].fillna('')

tfv_matrix = tfv.fit_transform(dataset['Summary'])

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(dataset.index, index=dataset['Title']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return dataset['Title'].iloc[movie_indices]

@app.route('/getrecs/', methods=['GET'])
def respond():
    # Retrieve the title from url parameter
    title = request.args.get("title", None)

    # For debugging
    print(f"got title {title}")

    response = {}

    if not title:
        response["ERROR"] = "no title found, please send a title."
    elif str(title).isdigit():
        response["ERROR"] = "title can't be numeric."
    else:
        response["recs"] = give_rec(title).to_json(orient="records")

    # Return the response in json format
    return Response(json.dumps(response), mimetype='application/json')

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)