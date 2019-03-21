"""accuracy.py

Usage:
  accuracy.py

"""

import warnings
import logging
import itertools
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, MultinomialHMM, GMMHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
from elasticsearch import helpers, Elasticsearch
import csv
from pandas.io.json import json_normalize
from datetime import datetime, timedelta
from objdict import ObjDict
import json
import datetime

INDEX_NAME = 'predictions'
TYPE_NAME = 'outcome'
TRADE_TYPE_NAME = 'accuracy'
TRADE_INDEX_NAME = 'accuracy'
ID_FIELD = 'date'

es = Elasticsearch()

class ESProxy(object):

    def delete_and_create_index(self):

        if es.indices.exists(INDEX_NAME):
            print("deleting '%s' index..." % (INDEX_NAME))
            res = es.indices.delete(index = INDEX_NAME)

        request_body = {
            "settings" : {
                "number_of_shards": 1,
                "number_of_replicas": 0
                }
        }
        print("creating '%s' index..." % (INDEX_NAME))
        res = es.indices.create(index = INDEX_NAME, body = request_body)

class AccuracyCalculator(object):

    def __init__(self, ticker, verbose=False, prediction_date=None):

        self.verbose = verbose
        self.ticker = ticker
        self.prediction_date = prediction_date
        self.fetch_data()
        self.calculate()

    def fetch_data(self):

        print("Fetching stock data ...")
        res = es.search(index="market", doc_type="quote", size=10000, body={"query": {"match": {"ticker": self.ticker}}})
        self.stock_data = json_normalize(res['hits']['hits'])

        print("Fetching recommendation data ...")
        res = es.search(index="recommendation", doc_type="trades", size=10000, body={"query": {"match": {"ticker": self.ticker}}})
        self.recommendation_data = json_normalize(res['hits']['hits'])

    def json_data_for_accuracy(self):

        rows = list()

        # meta
        ticker = self.ticker
        date = self.prediction_date
        prediction = self.prediction
        id = "%s-%s-%s" % (ticker, date, prediction)

        meta = {
            "index": {
                "_index": TRADE_INDEX_NAME,
                "_type": TRADE_TYPE_NAME,
                "_id": id
            }
        }
        rows.append(json.dumps(meta))

        # data
        row = ObjDict()
        row.result = self.result
        row.prediction = self.prediction
        row.prediction_date = self.prediction_date
        row.ticker = self.ticker
        row.accuracy = self.accuracy
        rows.append(json.dumps(row))

        return rows

    def calculate(self):

        accuracy_data = list()
        accuracy = False

        #print("total score for %s was %s, close that day was %s" % (self.ticker, self.recommendation_data, self.stock_data))
        result = self.stock_data.loc[self.stock_data['_source.timestamp'] == self.prediction_date].tail(1)['_source.change'].values[0]
        prediction = self.recommendation_data.loc[self.recommendation_data['_source.timestamp'] == self.prediction_date].tail(1)['_source.total_score'].values[0]
        if result > 0 and prediction > 0: accuracy = True
        if result < 0 and prediction < 0: accuracy = True

        self.result = result
        self.prediction = prediction
        self.accuracy = accuracy
        print("accuracy:%s score %s result %s" % (accuracy, prediction, result))

        accuracy_rows = self.json_data_for_accuracy()
        accuracy_data.append(accuracy_rows)

        print("Exporting accuracy to ES")
        es_array = self.format_data_for_es(accuracy_data)
        res = es.bulk(index = TRADE_INDEX_NAME, body = es_array, refresh = True)

    def format_data_for_es(self, data):
        es_array = ""
        for row in data:
            es_array += row[0]
            es_array += "\n"
            es_array += row[1]
            es_array += "\n"
        return es_array


def delete_all_accuracy_data():
   print("Deleting all accuracy data")
   es.delete_by_query(index="accuracy",doc_type="accuracy", body={'query': {'match_all': {}}})

if __name__ == '__main__':
    # "2019-01-02T06:00:00"

    delete_all_accuracy_data()
    print("Fetching dates ...")
    res = es.search(index="predictions", doc_type="outcome", size=10000, body={"query": { "range" : { "timestamp" : { "gte" : "2019-03-10T06:00:00"}}}})
    rows = json_normalize(res['hits']['hits'])
    print(rows)

    for row in rows.iterrows():
        actual_date = row[1]['_source.timestamp']
        ticker = row[1]['_source.ticker']
        print("calculating accuracy for %s on %s" % (ticker, actual_date))
        accuracy_calculator = AccuracyCalculator(ticker=ticker, verbose=True, prediction_date = actual_date)
        accuracy_calculator.calculate()
