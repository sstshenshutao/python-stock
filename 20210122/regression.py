import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq


def nonlinear_approx(x_input, fx, L, e):
    if (len(x_input.shape)) == 1:
        x_input = x_input[:, np.newaxis]
    # choose L random elements of the data
    rng = np.random.default_rng()
    x_l = rng.choice(x_input, L)
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


def plot_regression(points, center_number=3, err_value=0.4, predict_x=None):
    # plot the ori
    plt.figure(figsize=(10, 10))
    plt.plot(points[:, 0], points[:, 1], label='original')
    # plot the regression
    regression_func, _, _, epsilon, _ = nonlinear_approx(points[:, 0], points[:, 1], center_number, err_value)
    regression_y = regression_func(points[:, 0])
    plt.plot(points[:, 0], regression_y, label='regression')
    # predict
    predict_y = None
    if predict_x is not None:
        predict_y = regression_func(predict_x)
        plt.plot(predict_x, predict_y, label='predict')
    plt.legend()
    plt.savefig('reg_%s_%d.png' % (filename[:-4], center_number), dpi=300)
    plt.close()
    return predict_y


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
        today = "2021_01_22"
        filename = '%s_%s.csv' % (stock, today)
        df = pd.read_csv(filename, index_col=0)
        day_length = 90
        predict_day = 1
        center_param_plot_y = []
        center_param_plot_x = []
        for param_num in range(day_length // 10, day_length // 2, 1):
            print("doing: %d" % param_num)
            center_parameter = param_num
            to_points = prepare_data(df, analysis_domain=day_length)
            predict_y = plot_regression(to_points, center_number=center_parameter, err_value=0.5,
                                        predict_x=np.array(range(day_length, day_length + predict_day, 1)))
            if predict_y is not None:
                center_param_plot_y.append(predict_y[0])
                center_param_plot_x.append(center_parameter)
        plt.figure(figsize=(10, 10))
        plt.plot(center_param_plot_x, center_param_plot_y)
        plt.savefig('center_param_plot_%s.png' % (filename[:-4]), dpi=300)
