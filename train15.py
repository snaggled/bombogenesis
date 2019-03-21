"""train.py

Usage:
  train.py <ticker> <date>

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
TRADE_TYPE_NAME = 'trades'
TRADE_INDEX_NAME = 'recommendation'
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

class StockPredictor(object):

    def __init__(self, ticker, chunks = 9, delta = 0, n_hidden_states=5, n_latency_days=10, n_steps_frac_change=10, n_steps_frac_high=30, n_steps_frac_low=10, n_iter=100, verbose=False, prediction_date=None):

        self.total_score = 0
        self.verbose = verbose
        self.ticker = ticker
        self.n_latency_days = n_latency_days
        self.hmm = GMMHMM(n_components=n_hidden_states, n_iter=n_iter)
        self.chunks = chunks
        self.delta = delta
        self.prediction_date = prediction_date
        self.fetch_training_data()
        self._compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def fetch_training_data(self):

        print("Fetching training data ...")
        res = es.search(index="market", doc_type="quote", size=10000, body={"query": {"match": {"ticker": self.ticker}}})
        self.training_data = json_normalize(res['hits']['hits'])
        self.chunked_training_data = self.training_data

        #vectors = []
        #chunked_training_data_lengths = []
        #start_index = 0
        #end_index = start_index + self.chunks
        #delta_date_index = end_index + self.delta

        #while delta_date_index <= len(self.training_data):
        #training_chunk = self.training_data[start_index:end_index]
        #    delta_chunk = self.training_data.iloc[delta_date_index]
        #    total_chunk = training_chunk.append(delta_chunk)
        #    #print("%s training_chunk to train %s" % (total_chunk, self.ticker))
        #    start_index = end_index + 1
        #    end_index = start_index + self.chunks
        #    delta_date_index = end_index + self.delta
        #    vectors.append(total_chunk)
        #    chunked_training_data_lengths.append(len(total_chunk))
        #    if self.verbose: print(total_chunk)

        #self.chunked_training_data = pd.DataFrame(np.concatenate(vectors), columns = self.training_data.columns)
        #self.chunked_training_data_lengths = chunked_training_data_lengths

        if self.verbose: print("Latest record for training:\n%s" % self.chunked_training_data.tail(1))
        latest_date = self.chunked_training_data.tail(1)['_source.timestamp']
        datetime_object = datetime.datetime.strptime(latest_date.values[0], '%Y-%m-%dT%H:%M:%S')

        if self.prediction_date == None:
            prediction_date = datetime_object + timedelta(days=self.delta + 1)
            self.prediction_date = datetime.datetime.strftime(prediction_date, '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def _extract_features(data):

        frac_change = np.array(data['_source.change']) #(close_price - open_price) / open_price
        frac_high = np.array(data['_source.change_high']) #(high_price - open_price) / open_price
        frac_low = np.array(data['_source.change_low']) #(open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        print('Extracting Features')
        feature_vector = StockPredictor._extract_features(self.chunked_training_data)
        if self.verbose: print("feature vector %s" % feature_vector)
        print('Training Model with %s features' % feature_vector.size)
        print("Latest date to be used in training is %s" % self.chunked_training_data.tail(1)['_source.timestamp'].values[0])
        #self.hmm.fit(feature_vector, self.chunked_training_data_lengths)
        self.hmm.fit(feature_vector)
        print('Model trained')

    def _compute_all_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.05, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.05, n_steps_frac_low)

        self.all_possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def json_data_for_trade(self):

        rows = list()

        # meta
        ticker = self.ticker
        date = self.prediction_date
        total_score = self.total_score
        id = "%s-%s-%s" % (ticker, date, total_score)

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
        row.total_score = total_score
        row.timestamp = self.prediction_date
        row.ticker = self.ticker
        rows.append(json.dumps(row))

        return rows

    def json_data_for_outcome(self, outcome, score):

        rows = list()

        # meta
        ticker = self.ticker
        date = self.prediction_date
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
        open_price = self.training_data.tail(1)['_source.open'].values[0]
        predicted_close = open_price * (1 + outcome[0])
        expected_value = outcome[0] * score
        row.predicted_close = predicted_close
        row.expected_value = expected_value
        row.timestamp = self.prediction_date
        row.score = score
        row.chunks = self.chunks
        row.delta = self.delta
        row.score = score
        row.ticker = self.ticker
        rows.append(json.dumps(row))

        return rows

    def delete_prediction_data(self, ticker):
        print("Deleting prediction data for ... %s" % self.ticker)
        es.delete_by_query(index=INDEX_NAME,doc_type=TYPE_NAME, body={'query': {'match': {'ticker': self.ticker}}})

    def predict_outcomes(self):

        print("predicting outcomes for: %s" % self.prediction_date)
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
        trade_data = list()
        outcome_score = []

        for possible_outcome in self.all_possible_outcomes:

            test_feature_vectors = np.row_stack((test_data_features, possible_outcome))
            score = self.hmm.score(test_feature_vectors)

            # ignoring scores <= 0
            if score > 0:
                rows = self.json_data_for_outcome(possible_outcome, score)
                bulk_data.append(rows)

                if possible_outcome[0] > 0:
                    self.total_score = self.total_score + score
                if possible_outcome[0] < 0:
                    self.total_score = self.total_score - score
                trade_rows = self.json_data_for_trade()
                trade_data.append(trade_rows)

        print("Exporting predictions to ES")

        es_array = self.format_data_for_es(bulk_data)
        res = es.bulk(index = INDEX_NAME, body = es_array, refresh = True)

        es_array = self.format_data_for_es(trade_data)
        res = es.bulk(index = TRADE_INDEX_NAME, body = es_array, refresh = True)

    def format_data_for_es(self, data):
        es_array = ""
        for row in data:
            es_array += row[0]
            es_array += "\n"
            es_array += row[1]
            es_array += "\n"
        return es_array

def delete_all_prediction_data():
   print("Deleting all PREDICTION data")
   es.delete_by_query(index=INDEX_NAME,doc_type=TYPE_NAME, body={'query': {'match_all': {}}})

if __name__ == '__main__':
    arguments = docopt(__doc__, version='train 0.1')
    ticker = arguments['<ticker>']
    date = arguments['<date>']

    if ticker == "ALL":
        print("Training all models")

        if date == "ALL":
            delete_all_prediction_data()
            print("Fetching dates ...")
            res = es.search(index="market", doc_type="quote", size=10000, body={"query": { "range" : { "timestamp" : { "gte" : "2019-03-10T06:00:00"}}}})
            rows = json_normalize(res['hits']['hits'])
            print(rows)
            for row in rows.iterrows():
                try:
                    actual_date = row[1]['_source.timestamp']
                    ticker = row[1]['_source.ticker']

                    print("predicting %s on %s" % (ticker, actual_date))
                    stock_predictor = StockPredictor(ticker=ticker, verbose=False, chunks=9, delta = 0, prediction_date = actual_date)
                    stock_predictor.fit()
                    stock_predictor.predict_outcomes()

                except:
                    print("Failed to train models for %s" % ticker)
    else:

        stock_predictor = StockPredictor(ticker=ticker, verbose=False, chunks=9, delta = 0, prediction_date = date)
        #stock_predictor.delete_prediction_data(ticker)
        stock_predictor.fit()
        stock_predictor.predict_outcomes()
