import yfinance as yf
import json
from datetime import date

today = date.today().strftime('%Y_%m_%d')


def query_save(stock_code):
    # 构造一个对象
    msft = yf.Ticker(stock_code)
    # 打印此对象信息
    with open("query_date_%s.json" % today, 'w') as fp:
        json.dump(msft.info, fp)
    # 返回 全部历史信息  Date	Open	High	Low	Close	Volume(成交股数)	Dividends(股息)	Stock Splits(拆股 default0)
    stock_history = msft.history(period="max")

    newest_date = stock_history.tail(1).index.to_list()[0].strftime('%Y_%m_%d')
    stock_history.to_csv('%s_%s.csv' % (stock_code, newest_date))


if __name__ == '__main__':
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'EA', 'NTES']
    for stock in stocks:
        query_save(stock)
