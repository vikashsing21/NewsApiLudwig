from flask import Flask, request, jsonify
from ludwig.api import LudwigModel
from newsapi import NewsApiClient
import pandas as pd
import json

app = Flask(__name__)
model = LudwigModel.load('./results/api_experiment_run/model')

newsapi = NewsApiClient(api_key='3bff10bbdc444b09bbe7ddf8e6b381e3')


def results(news):
    df_data = pd.read_json(json.dumps(news))
    df_pred = model.predict(data_df=df_data)
    # combining news data with predicted result
    df_combine = pd.concat(
        [df_data, df_pred['news_sentiment_predictions']], axis=1)
    # now formatting this result in proper order similar to newsapi structure
    df_temp = pd.DataFrame(df_combine)
    responseData = json.loads(df_temp.to_json(orient="records"))
    return responseData


def topGen():
    top_headlines_general = newsapi.get_top_headlines(category='general',
                                                      language='en',
                                                      page_size=100,
                                                      country='in')
    return results(top_headlines_general['articles'])


def topEnt():
    top_headlines_entertainment = newsapi.get_top_headlines(category='entertainment',
                                                            language='en',
                                                            page_size=100,
                                                            country='in')
    return results(top_headlines_entertainment['articles'])


def topTech():
    top_headlines_technology = newsapi.get_top_headlines(category='technology',
                                                         language='en',
                                                         page_size=100,
                                                         country='in')
    return results(top_headlines_technology['articles'])


def topScience():
    top_headlines_science = newsapi.get_top_headlines(category='science',
                                                      language='en',
                                                      page_size=100,
                                                      country='in')
    return results(top_headlines_science['articles'])


def topSports():
    top_headlines_sports = newsapi.get_top_headlines(category='sports',
                                                     language='en',
                                                     page_size=100,
                                                     country='in')
    return results(top_headlines_sports['articles'])


def topHealth():
    top_headlines_health = newsapi.get_top_headlines(category='health',
                                                     language='en',
                                                     page_size=100,
                                                     country='in')
    return results(top_headlines_health['articles'])

def topBusiness():
    top_headlines_business = newsapi.get_top_headlines(category='business',
                                                     language='en',
                                                     page_size=100,
                                                     country='in')
    return results(top_headlines_business['articles'])


def positivenews(newsData):
    posdata = []
    for data in newsData:
        if data['news_sentiment_predictions'] == 'positive' or data['news_sentiment_predictions'] == 'neutral':
            posdata.append(data)
    return posdata


def negativenews(newsData):
    negdata = []
    for data in newsData:
        if data['news_sentiment_predictions'] == 'negative':
            negdata.append(data)
    return negdata

# General
@app.route('/predict/general/positive', methods=['GET'])
def predictGenPos():
    response = positivenews(topGen())
    return jsonify(response)

@app.route('/predict/general/negative', methods=['GET'])
def predictGenNeg():
    response = negativenews(topGen())
    return jsonify(response)

# Health
@app.route('/predict/health/positive', methods=['GET'])
def predictHealthPos():
    response = positivenews(topHealth())
    return jsonify(response)

@app.route('/predict/health/negative', methods=['GET'])
def predictHealthNeg():
    response = negativenews(topHealth())
    return jsonify(response)


@app.route('/predict/technology', methods=['GET'])
def predictTech():
    response = topTech()
    return jsonify(response)


@app.route('/predict/science', methods=['GET'])
def predictScience():
    response = topScience()
    return jsonify(response)


@app.route('/predict/entertainment', methods=['GET'])
def predictEnt():
    response = topEnt()
    return jsonify(response)


@app.route('/predict/sports', methods=['GET'])
def predictSports():
    response = topSports()
    return jsonify(response)


@app.route('/predict/business', methods=['GET'])
def predictBusiness():
    response = topBusiness()
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)

#     spacy==2.2.3
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5
