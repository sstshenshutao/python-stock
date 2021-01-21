import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from datetime import date


def strategy(slice_ori, outlier_factor=1.2, day_length=7):
    out_money = 0
    stock_value = 0
    # 在7天时段内的过低值时买入,并且在过高时卖出(卖空),观察收益
    day_start = 0
    day_end = day_start + day_length
    day_len = slice_ori.index.size
    buy_sell_time = 0
    while day_end < day_len - 1:
        current_upper_bound, current_lower_bound = cal_bound(slice_ori[day_start:day_end], outlier_factor)
        today_price = slice_ori[day_end:day_end + 1]['Close'].to_list()[0]
        today_date = slice_ori[day_end:day_end + 1].index.to_list()[0]
        if today_price > current_upper_bound['Close']:
            # 賣空
            out_money += today_price
            stock_value -= 1
            print("买入: 在价格%f, %s" % (today_price, today_date))
            buy_sell_time += 1
        elif today_price < current_lower_bound['Close']:
            # 買入
            out_money -= today_price
            stock_value += 1
            print("卖出: 在价格%f, %s" % (today_price, today_date))
            buy_sell_time += 1
        day_start += 1
        day_end = day_start + day_length
    day_end_price = slice_ori[day_end:day_end + 1]['Close'].to_list()[0]
    day_end_date = slice_ori[day_end:day_end + 1].index.to_list()[0]
    print(
        "收益: %f, 持仓: %d, 股价: %f, 结算日期: %s, 交易次数: %d" % (
            stock_value * day_end_price + out_money, stock_value, out_money, day_end_date, buy_sell_time))
    return stock_value * day_end_price + out_money, buy_sell_time


def cal_bound(slice_ori, outlier_factor=1.2, day_length=7):
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
    # print(upper_bound, lower_bound)
    # 过高outlier
    # print(slice_ori[slice_ori['Open'] > upper_bound['Open']])
    # print(slice_ori[slice_ori['High'] > upper_bound['High']])
    # print(slice_ori[slice_ori['Low'] > upper_bound['Low']])
    # print(slice_ori[slice_ori['Close'] > upper_bound['Close']])
    # 过低outlier
    # print(slice_ori[slice_ori['Open'] < lower_bound['Open']])
    # print(slice_ori[slice_ori['High'] < lower_bound['High']])
    # print(slice_ori[slice_ori['Low'] < lower_bound['Low']])
    # print(slice_ori[slice_ori['Close'] < lower_bound['Close']])
    return [upper_bound, lower_bound]


def plot_stock(stock_code):
    # read the csv
    today = date.today().strftime('%Y_%m_%d')
    filename = '%s_%s.csv' % (stock_code, today)
    df = pd.read_csv(filename, index_col=0)
    total_length = 60
    day_length = 7
    # trim the table
    slice_ori = df.tail(total_length)
    # print(df.tail(3)['Close'])
    outlier_factor_samples = np.linspace(0.3, 1.5, 51, endpoint=True)
    # print(outlier_factor_samples)
    result_list = []
    trade_time = []
    for factor in outlier_factor_samples:
        result_list.append(strategy(slice_ori, outlier_factor=factor)[0])
        trade_time.append(strategy(slice_ori, outlier_factor=factor)[1])
    plt.figure(figsize=(10, 10))
    plt.plot(outlier_factor_samples, result_list)
    plt.savefig('%s_profit_%d.png' % (filename[:-4], day_length), dpi=300)
    plt.figure(figsize=(10, 10))
    plt.plot(outlier_factor_samples, trade_time)
    plt.savefig('%s_profit_time_%d.png' % (filename[:-4], day_length), dpi=300)


if __name__ == '__main__':
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'EA', 'NTES']
    for stock in stocks:
        plot_stock(stock)
    # plt.clf()
    # ax = plt.gca()
    # boxplot = slice_df.boxplot(ax=ax, column=['Open', 'High', 'Low', 'Close'])
    # plt.savefig('box_%d' % day_length, dpi=300)
