import numpy as np
from unittest import TestCase

# csv data格式 ： code,date,收盘,开盘, 最高,最低,成交量

class TestNumpyStock(TestCase):
    def test_read_file(self):
        file_name = "stock_data.csv"
        close_price, volume = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=(2, 6),
            unpack=True
        )
        print("闭盘价序列为", close_price)
        print("交易量序列为", volume)

    def test_max_and_min(self):
        high_price, low_price = np.loadtxt(
            fname="stock_data.csv",
            delimiter=",",
            usecols=(4, 5),
            unpack=True
        )
        print("最高价序列为", high_price)
        print("最低价序列为", low_price)

    def test_ptp_cal(self):
        """
        计算最高价极差和最低价极差
        """
        high_price, low_price = np.loadtxt(
            fname="stock_data.csv",
            delimiter=",",
            usecols=(4, 5),
            unpack=True
        )
        print("股票最高价的极差为", np.ptp(high_price))
        print("股票最低价的极差为", np.ptp(low_price))

    def test_avg_cal(self):
        """
        计算成交量加权平均价格VWAP(Volume Weighted Average)，代表着金融资产的平均价格(收盘价按照当日成交量取加权平均）
        """
        file_name = "stock_data.csv"
        close_price, volume = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=(2, 6),
            unpack=True
        )
        print("avg price(close_price) = {}".format(np.average(close_price)))
        print("VWAP(Volume Weighted Average Price) = {}".format(np.average(close_price, weights=volume)))

    def test_median(self):
        """
        计算收盘价的中位数
        """
        file_name = "stock_data.csv"
        close_price = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=2,
            unpack=True
        )
        print("median(close_price) = {}".format(np.median(close_price)))

    def test_var(self):
        """
        计算收盘价的方差
        """
        file_name = "stock_data.csv"
        close_price = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=2,
            unpack=True
        )
        print("var(close_price) = {}".format(np.var(close_price)))

    def test_volatility(self):
        """
        波动率是价格变动的一种度量，历史波动率可以根据历史数据计算得出。在计算的时候，需要用到对数波动率。
        年波动率等于对数收益率的标准差除以其均值，再乘以交易日的平方根，通常交易日取250天
        月波动率等于对数收益率的标准差除以其均值，再乘以交易月的平方根，通常交易月取12个月
        """
        file_name = "stock_data.csv"
        close_price, volume = np.loadtxt(
            fname=file_name,
            delimiter=",",
            usecols=(2, 6),
            unpack=True
        )
        log_return = np.diff(np.log(close_price))
        annual_volatility = log_return.std() / log_return.mean() * np.sqrt(250)
        monthly_volatility = log_return.std() / log_return.mean() * np.sqrt(12)

        print("log return = {}".format(log_return))
        print("annual volatility = {}".format(annual_volatility))
        print("monthly volatility = {}".format(monthly_volatility))


if __name__ == "__main__":
    obj = TestNumpyStock()
    obj.test_read_file()
    obj.test_max_and_min()
    obj.test_ptp_cal()
    obj.test_avg_cal()
    obj.test_median()
    obj.test_var()
    obj.test_volatility()
