import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, MultinomialHMM, GMMHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
from elasticsearch import helpers, Elasticsearch
import csv
from pandas.io.json import json_normalize
from datetime import datetime
from objdict import ObjDict
import json
import datetime

INDEX_NAME = 'predictions'
TYPE_NAME = 'outcome'
ID_FIELD = 'date'

query = '''{
"query": {
"bool": {
"filter": [
{
  "bool": {
    "filter": [
      {
        "bool": {
          "should": [
            {
              "match_phrase": {
                "ticker": "%s"
              }
            }
          ],
          "minimum_should_match": 1
        }
      },
      {
        "bool": {
          "should": [
            {
              "range": {
                "timestamp": {
                  "%s": "%s"
                }
              }
            }
          ],
          "minimum_should_match": 1
        }
      }
    ]
  }
}
],
"should": [],
"must_not": []
}}}'''

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

class StockPredictor(object):

    def __init__(self, ticker, n_hidden_states=5, n_latency_days=10, n_steps_frac_change=50, n_steps_frac_high=30, n_steps_frac_low=10, n_iter=1000, verbose=False):

        self.verbose = verbose
        self.ticker = ticker
        self.n_latency_days = n_latency_days

        self.hmm = GMMHMM(n_components=n_hidden_states, n_iter=n_iter)

        self.fetch_training_data()
        self.fetch_latest_data() # to predict

        self._compute_allall_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def fetch_latest_data(self):

        print("Fetching latest data ...")
        res = es.search(index="market", doc_type="quote", size=10000, body={"query": {"match": {"ticker": self.ticker}}})
        latest_data = json_normalize(res['hits']['hits'])
        self.latest_data = latest_data.tail(1)
        if self.verbose: print("Latest data:\n%s" % self.latest_data)

    def fetch_training_data(self):

        print("Fetching training data ...")
        res = es.search(index="market", doc_type="quote", size=10000, body={"query": {"match": {"ticker": self.ticker}}})
        self.training_data = json_normalize(res['hits']['hits'])
        self.training_data.drop(self.training_data.tail(1).index,inplace=True)
        print("%s records to train %s" % (len(self.training_data.index), self.ticker))
        if self.verbose: print("Latest record for training:\n%s" % self.training_data.tail(1))

        # tbd - to use es instead
        #q = query % (self.ticker, "lt", datetime.date.today().strftime("%Y-%m-%d"))
        #print(q)
        #res = es.search(index=INDEX_NAME, doc_type=TYPE_NAME, size=10000, body=query)

    @staticmethod
    def _extract_features(data):

        frac_change = np.array(data['_source.change']) #(close_price - open_price) / open_price
        frac_high = np.array(data['_source.change_high']) #(high_price - open_price) / open_price
        frac_low = np.array(data['_source.change_low']) #(open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        print('Extracting Features')
        feature_vector = StockPredictor._extract_features(self.training_data)
        if self.verbose: print("feature vector %s" % feature_vector)
        print('Training Model with %s features' % feature_vector.size)
        print("Latest date to be used in training is %s" % self.training_data.tail(1)['_source.timestamp'].values[0])
        self.hmm.fit(feature_vector)
        print('Model trained')

    def _compute_allall_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)

        self.all_possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def json_data_for_outcome(self, day, outcome, score):

        rows = list()

        # meta
        ticker = day['_source.ticker']
        date = day['_source.timestamp']
        vector = outcome
        id = "%s-%s-%s" % (ticker, date, vector)

        meta = {
            "index": {
                "_index": INDEX_NAME,
                "_type": TYPE_NAME,
                "_id": id
            }
        }
        rows.append(json.dumps(meta))

        # data
        row = ObjDict()
        row.frac_change = outcome[0]
        row.frac_high_range = outcome[1]
        row.frac_low_range = outcome[2]
        open_price = day['_source.open'].values[0]
        predicted_close = open_price * (1 + outcome[0])
        expected_value = outcome[0] * score
        row.predicted_close = predicted_close
        row.expected_value = expected_value
        row.timestamp = day['_source.timestamp'].values[0]
        row.score = score
        row.ticker = day['_source.ticker'].values[0]
        rows.append(json.dumps(row))

        return rows

    def predict_outcomes(self):

        print("predicting outcomes for: %s" % self.latest_data['_source.timestamp'].values[0])
        previous_testing_data = self.training_data.tail(self.n_latency_days).index

        if self.verbose:
            print("previous_testing_data %s" % previous_testing_data)

        test_data = self.training_data.iloc[previous_testing_data]

        if self.verbose:
            print("Using the following slice of data:")
            print("[%s]" % previous_testing_data)
            print(test_data)

        test_data_features = StockPredictor._extract_features(test_data)

        # to blow everything away - may need to recreate/refresh indexes in ES!
        #self.delete_and_create_index()

        bulk_data = list()
        outcome_score = []

        for possible_outcome in self.all_possible_outcomes:

            test_feature_vectors = np.row_stack((test_data_features, possible_outcome))

            if self.verbose:
                print("Final test feature set:")
                print("[%s]" % test_feature_vectors)

            score = self.hmm.score(test_feature_vectors)

            # ignoring scores <= 0
            if score > 0:
                rows = self.json_data_for_outcome(self.latest_data, possible_outcome, score)
                bulk_data.append(rows)

        # format for ES, ugly
        es_array = ""
        for row in bulk_data:
            es_array += row[0]
            es_array += "\n"
            es_array += row[1]
            es_array += "\n"

        #print("Deleting prediction data for ... %s" % day['_source.ticker'])
        #es.delete_by_query(index=INDEX_NAME,doc_type=TYPE_NAME, body={'query': {'match': {'ticker': day['_source.ticker']}}})

        print("Exporting predictions to ES")
        if self.verbose: print(es_array)
        res = es.bulk(index = INDEX_NAME, body = es_array, refresh = True)

if __name__ == '__main__':
    with open('nasdaq100list.csv', 'r') as f:
        reader = csv.reader(f)
        stocks = list(reader)
        for stock in stocks:
            ticker = stock[0]
            if ticker == "Symbol": continue
            try:
                stock_predictor = StockPredictor(ticker=ticker, verbose=False)
                stock_predictor.fit()
                stock_predictor.predict_outcomes()
            except:
                print("Failed to train models for %s" % ticker)
