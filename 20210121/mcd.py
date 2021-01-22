from stock_query import query_save
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from datetime import date


def cal_bound(slice_ori, outlier_factor=0.3):
    slice_df = slice_ori[['Open', 'High', 'Low', 'Close']]
    # print(datetime.strptime(df.tail(1).index.to_list()[0], '%Y-%m-%d'))
    quantil = slice_df.quantile([0.25, 0.5, 0.75])
    # print(quantil)
    iqr = {'Open': quantil.loc[0.75, 'Open'] - quantil.loc[0.25, 'Open'],
           'High': quantil.loc[0.75, 'High'] - quantil.loc[0.25, 'High'],
           'Low': quantil.loc[0.75, 'Low'] - quantil.loc[0.25, 'Low'],
           'Close': quantil.loc[0.75, 'Close'] - quantil.loc[0.25, 'Close']}
    upper_bound = {'Open': quantil.loc[0.75, 'Open'] + outlier_factor * iqr['Open'],
                   'High': quantil.loc[0.75, 'High'] + outlier_factor * iqr['High'],
                   'Low': quantil.loc[0.75, 'Low'] + outlier_factor * iqr['Low'],
                   'Close': quantil.loc[0.75, 'Close'] + outlier_factor * iqr['Close']}
    lower_bound = {'Open': quantil.loc[0.25, 'Open'] - outlier_factor * iqr['Open'],
                   'High': quantil.loc[0.25, 'High'] - outlier_factor * iqr['High'],
                   'Low': quantil.loc[0.25, 'Low'] - outlier_factor * iqr['Low'],
                   'Close': quantil.loc[0.25, 'Close'] - outlier_factor * iqr['Close']}
    print(slice_ori.index.size, lower_bound['Close'], upper_bound['Close'])
    return [upper_bound, lower_bound]


if __name__ == '__main__':
    stocks = ['MCD']
    for stock in stocks:
        query_save(stock)
        today = date.today().strftime('%Y_%m_%d')
        filename = '%s_%s.csv' % (stock, today)
        df = pd.read_csv(filename, index_col=0)
        ll = []
        uu = []
        # print(df.tail(5))
        for day_time in range(5, 15):
            result = cal_bound(df.tail(day_time), outlier_factor=0.3)
            ll.append(result[0]['Close'])
            uu.append(result[1]['Close'])
        plt.figure(figsize=(10, 10))
        plt.plot(range(5, 15), ll)
        plt.savefig('ll%s.png' % (filename[:-4]), dpi=300)
        plt.figure(figsize=(10, 10))
        plt.plot(range(5, 15), uu)
        plt.savefig('uu%s.png' % (filename[:-4]), dpi=300)
