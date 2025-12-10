from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
from unittest import TestCase

class TestNumpyMA(TestCase):
    def test_sma(self):
        file_name = "stock_data.csv"
        close_price = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=(2),
            unpack=True
        )
        # 设置卷积窗口,权重序列
        N = 5   # 五日均线
        weights = np.ones(N)/N
        print("the close price is",close_price)
        sma_close_price = np.convolve(weights, close_price)[N-1:-N+1]
        print("SMA result of the close price is",sma_close_price)

        # plot the graph
        plt.figure(1)
        plt.plot(sma_close_price)
        plt.show()

        plt.figure(2)
        plt.plot(close_price)
        plt.show()
    def test_ema(self):
        """
        指数加权平均算法：https://blog.csdn.net/LiuHDme/article/details/104744836
        """
        file_name = "stock_data.csv"
        close_price = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=(2),
            unpack=True
        )
        # 设置卷积窗口,权重序列
        N = 5  # 五日均线
        weights = np.linspace(-1,0,N)
        weights = weights/np.sum(weights)
        ema_close_price = np.convolve(weights, close_price)[N - 1:-N + 1]
        print("EMA result of the close price is", ema_close_price)
        # plot the graph
        plt.figure(3)
        plt.plot(ema_close_price)
        plt.show()






if __name__ == '__main__':
    TestNumpyMA().test_sma()
    TestNumpyMA().test_ema()

