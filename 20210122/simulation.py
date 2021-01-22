import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
from datetime import date


def strategy(df_ori, current_delta=0.2):
    out_money = 0
    stock_value = 0
    buy_sell_time = 0
    # 模拟策略60天 [60-1]
    day_offset = 60
    whole_size = df_ori.index.size

    while day_offset > 0:
        unitl_some_day = df_ori[:whole_size - day_offset]
        current_upper_bound, current_lower_bound = regulated_bound(unitl_some_day, delta=current_delta)
        # the next day
        today_price = df_ori[whole_size - day_offset:whole_size - day_offset + 1]['Close'].to_list()[0]
        today_date = df_ori[whole_size - day_offset:whole_size - day_offset + 1].index.to_list()[0]
        # print("[今天]: %f, %s" % (today_price, today_date))
        # print(current_upper_bound, current_lower_bound)
        if today_price > current_upper_bound:
            # 賣空
            out_money += today_price
            stock_value -= 1
            print("卖出: 在价格%f, %s, %f" % (today_price, today_date, current_upper_bound))
            buy_sell_time += 1
        elif today_price < current_lower_bound:
            # 買入
            out_money -= today_price
            stock_value += 1
            print("买入: 在价格%f, %s, %f" % (today_price, today_date, current_lower_bound))
            buy_sell_time += 1
        day_offset -= 1
    day_end_price = df_ori[whole_size - 1:whole_size]['Close'].to_list()[0]
    day_end_date = df_ori[whole_size - 1:whole_size].index.to_list()[0]
    print(
        "收益: %f, 持仓: %d, 账户: %f, 结算日期: %s, 交易次数: %d" % (
            stock_value * day_end_price + out_money, stock_value, out_money, day_end_date, buy_sell_time))
    return stock_value * day_end_price + out_money, buy_sell_time


def cal_regression_mean(until_someday_df):
    day_length = 90
    predict_day = 1
    center_param_plot_y = []
    center_param_plot_x = []
    for param_num in range(day_length // 10, day_length // 2, 1):
        # print("doing: %d" % param_num)
        center_parameter = param_num
        to_points = prepare_data(until_someday_df, analysis_domain=day_length)
        predict_y = regression(to_points, center_number=center_parameter, err_value=0.5,
                               predict_x=np.array(range(day_length, day_length + predict_day, 1)))
        if predict_y is not None:
            center_param_plot_y.append(predict_y[0])
            center_param_plot_x.append(center_parameter)
    df_regression = pd.DataFrame(zip(center_param_plot_x, center_param_plot_y))
    quantil = df_regression.quantile([0.25, 0.5, 0.75])
    mean = df_regression.mean().loc[1]
    return mean


def cal_box_bound_mean(until_someday_df, stock_outlier_factor):
    ll = []
    uu = []
    for day_time in range(5, 15):
        result = cal_bound(until_someday_df.tail(day_time), outlier_factor=stock_outlier_factor)
        uu.append(result[0]['Close'])
        ll.append(result[1]['Close'])
    lower_bound_mean = pd.DataFrame(ll).mean().loc[0]
    upper_bound_mean = pd.DataFrame(uu).mean().loc[0]
    return upper_bound_mean, lower_bound_mean


def regulated_bound(until_someday_df, delta=0.3, stock_outlier_factor=0.5):
    # regression
    mean = cal_regression_mean(until_someday_df)
    # box bound
    upper_bound_mean, lower_bound_mean = cal_box_bound_mean(until_someday_df, stock_outlier_factor)
    avg_bound = (lower_bound_mean + upper_bound_mean) / 2
    # 修正值
    regulation = delta * (mean - avg_bound)
    return upper_bound_mean + regulation, lower_bound_mean + regulation


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
    # print(slice_ori.index.size, lower_bound['Close'], upper_bound['Close'])
    return [upper_bound, lower_bound]


def nonlinear_approx(x_input, fx, L, e):
    if (len(x_input.shape)) == 1:
        x_input = x_input[:, np.newaxis]
    # choose L random elements of the data
    rng = np.random.default_rng()
    x_l = rng.choice(x_input, L - 3)
    x_l = np.vstack([x_l, [x_input[-1], x_input[-2], x_input[-3]]])
    # print(x_l)
    dij = cdist(x_input, x_l)
    # compute epsilon similar to diffusion maps
    epsilon = e * np.max(dij)
    phi_l = np.exp(-dij ** 2 / epsilon ** 2)
    a, res, _, _ = lstsq(phi_l, fx, cond=None)
    # augment phi_l
    phi_l = np.column_stack([phi_l, np.ones(phi_l.shape[0])])
    cb, res, _, _ = lstsq(phi_l, fx, cond=None)
    a, b = cb[:phi_l.shape[1] - 1], cb[phi_l.shape[1] - 1]

    def ret(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        ret_dij = cdist(x, x_l)
        ret_phi = np.exp(-ret_dij ** 2 / epsilon ** 2)
        return ret_phi @ a + b

    return ret, res, a, epsilon, x_l


def regression(points, center_number=3, err_value=0.4, predict_x=None):
    # plot the regression
    regression_func, _, _, epsilon, _ = nonlinear_approx(points[:, 0], points[:, 1], center_number, err_value)
    # predict
    return regression_func(predict_x)


def prepare_data(df_all, analysis_domain=5):
    # 取对应区间的值
    close_list = df_all.tail(analysis_domain)['Close'].to_list()
    points = np.ravel(np.column_stack((np.array(range(analysis_domain)), np.array(close_list)))).reshape(
        analysis_domain, 2)
    # print(points)
    return points


if __name__ == '__main__':
    stocks = ['MCD']
    for stock in stocks:
        # today = date.today().strftime('%Y_%m_%d')
        today = date.today().strftime('%Y_%m_%d')
        filename = '%s_%s.csv' % (stock, today)
        df = pd.read_csv(filename, index_col=0)
        delta_samples = np.linspace(0., 1., 11, endpoint=True)
        result_list = []
        trade_time = []
        for delta in delta_samples:
            print(delta)
            result = strategy(df, current_delta=delta)
            result_list.append(result[0])
            trade_time.append(result[1])
        plt.figure(figsize=(10, 10))
        plt.plot(delta_samples, result_list)
        plt.savefig('%s_profit.png' % (filename[:-4]), dpi=300)
        plt.figure(figsize=(10, 10))
        plt.plot(delta_samples, trade_time)
        plt.savefig('%s_profit_time.png' % (filename[:-4]), dpi=300)
