import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_curve(msft_stock_history, start='2020-01-01', title='msft_2020_01_01_curve'):
    plt.clf()
    msft_stock_history = msft_stock_history.copy()
    msft_stock_history = msft_stock_history[(msft_stock_history['Date'] > '2020-01-01')]
    # print(msft_stock_history.head())
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))

    # plot four curves
    plt.plot(msft_stock_history['Date'], msft_stock_history['Open'], label='open')
    plt.plot(msft_stock_history['Date'], msft_stock_history['High'], label='High')
    plt.plot(msft_stock_history['Date'], msft_stock_history['Low'], label='Low')
    plt.plot(msft_stock_history['Date'], msft_stock_history['Close'], label='Close')

    #
    plt.suptitle(title)
    plt.legend()
    plt.gcf().autofmt_xdate()

    # save the fig
    plt.savefig('%s.png' % title)


def plot_buy_open_sell_close(msft_stock_history, start='2020-01-01', title='msft_2020_01_01_buy_open_sell_close'):
    plt.clf()
    msft_stock_history = msft_stock_history.copy()
    msft_stock_history = msft_stock_history[(msft_stock_history['Date'] > '2020-01-01')]
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))

    msft_stock_history['diff'] = msft_stock_history['Close'] - msft_stock_history['Open']
    plt.plot(msft_stock_history['Date'], msft_stock_history['diff'], label='diff')
    plt.suptitle('%s: %f' % (title, msft_stock_history['diff'].sum()))
    plt.legend()
    plt.gcf().autofmt_xdate()

    # save the fig
    plt.savefig('%s.png' % title)


def plot_buy_yesterday_close_sell_open(msft_stock_history, start='2020-01-01',
                                       title='msft_2020_01_01_buy_yesterday_close_sell_open'):
    plt.clf()
    msft_stock_history = msft_stock_history.copy()
    msft_stock_history = msft_stock_history[(msft_stock_history['Date'] > '2020-01-01')]
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))

    msft_stock_history['Open_1'] = msft_stock_history['Open'].shift(1)
    msft_stock_history['diff'] = msft_stock_history['Open_1'] - msft_stock_history['Close']
    plt.plot(msft_stock_history['Date'], msft_stock_history['diff'], label='diff')
    plt.suptitle('%s: %f' % (title, msft_stock_history['diff'].sum()))
    plt.legend()
    plt.gcf().autofmt_xdate()

    # save the fig
    plt.savefig('%s.png' % title)


if __name__ == '__main__':
    # read the csv
    df = pd.read_csv('MSFT_2021_01_08.csv')
    plot_curve(df)
    plot_buy_yesterday_close_sell_open(df)
    plot_buy_open_sell_close(df)
