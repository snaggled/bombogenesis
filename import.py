"""import.py

Usage:
  import.py <ticker>

"""

from elasticsearch import helpers, Elasticsearch
import csv
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import json
from objdict import ObjDict
import datetime
import time
from hmmlearn.hmm import GaussianHMM
import numpy as np
import itertools
from docopt import docopt

INDEX_NAME = 'market'
TYPE_NAME = 'quote'
ID_FIELD = 'date'

es = Elasticsearch()

class Importer(object):

    def __init__(self, ticker):
        self.ticker = ticker

    def es_date_format(self, time):
        return time.strftime('%Y-%m-%dT06:00:00')

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

    def json_data_for_ticker(self, ticker, datasource, start_date, end_date):
        rows = list()
        dp = data.DataReader(ticker, datasource, start_date, end_date)
        for tuple in dp.iterrows():

            date = int(time.mktime(tuple[0].timetuple()))
            id = "%s-%s" % (date, ticker)
            meta = {
                "index": {
                        "_index": INDEX_NAME,
                        "_type": TYPE_NAME,
                        "_id": id
                    }
                }
            rows.append(meta)

            row = ObjDict()


            row.timestamp= self.es_date_format(tuple[0])
            row._timestamp= self.es_date_format(tuple[0])
            row.adjusted_close = tuple[1][5]
            row.ticker = ticker

            row.volume = tuple[1][4]
            open_price = tuple[1][2]
            close_price = tuple[1][3]
            high_price = tuple[1][0]
            low_price = tuple[1][1]
            row.open = open_price
            row.close = close_price
            row.high = high_price
            row.low = low_price
            row.change = (close_price - open_price) / open_price
            row.change_high = (high_price - open_price) / open_price
            row.change_low = (open_price - low_price) / open_price

            rows.append(json.dumps(row))

        return rows

    def fetch_data(self):
        start_date = '1990-01-01'
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        datasource = "yahoo"
        print("fetching data for %s between %s and %s" % (self.ticker, start_date, end_date))
        data = self.json_data_for_ticker(self.ticker, datasource, start_date, end_date)
        self.data = data

    def import_data(self):
        es_array = ""
        for row in self.data:
            es_array += str(row)
            es_array += "\n"
        print("importing ... %s" % data)
        print("importing %s records to ES" % len(self.data))

        res = es.bulk(index = INDEX_NAME, body = self.data, refresh = True)

#    arguments = docopt(__doc__, version='import 0.1')
#    ticker= arguments['<ticker>']

#    data = fetch_data(ticker)
#    import_data(data)

def delete_all_stock_data():
        print("Deleting all stock data")
        es.delete_by_query(index=INDEX_NAME,doc_type=TYPE_NAME, body={'query': {'match_all': {}}})

def import_ticker(ticker, delete=False):

    if delete:
        print("Deleting stock data for ... %s" % ticker)
        es.delete_by_query(index=INDEX_NAME,doc_type=TYPE_NAME, body={'query': {'match': {'ticker': ticker}}})

    importer = Importer(ticker)
    importer.fetch_data()
    importer.import_data()

#import_ticker("MDLZ", delete=True)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='import 0.1')
    ticker = arguments['<ticker>']
    if ticker == "ALL":
        delete_all_stock_data()
        print("Fetching all data")
        with open('sp500.csv', 'r') as f:
            reader = csv.reader(f)
            stocks = list(reader)
            for stock in stocks:
                ticker = stock[0]
                if ticker == "Symbol": continue
                try:
                    import_ticker(ticker, delete=False)
                except:
                    print("Failed to import %s" % ticker)
    else:
        import_ticker(ticker, delete=True)
