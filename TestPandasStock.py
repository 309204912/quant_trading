import pandas as pd
from unittest import TestCase

class TestPandasStock(TestCase):
    def test_read_file(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)

        print(df.info())
        print('-------------------')
        print(df.describe())

    def test_time(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month

        print(df)
    def test_close_min(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        print("the close min value is {}".format(df["Close"].min()))
        print("the close min index is {}".format(df["Close"].idxmin()))
        print("the close min row is\n{}".format(df.loc[df["Close"].idxmin()]))

    def test_mean(self):
        """
        计算每月平均收盘价与开盘价
        """
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        print("monthly average close price is {}".format(df.groupby("month")["Close"].mean()))
        print("monthly average open price is {}".format(df.groupby("month")["Open"].mean()))

    def test_ripples_ratio_cal(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["rise"] = df["Close"].diff()
        df["shifted_close"] = df["Close"].shift(1)
        df["ripples_ratio"] = df["rise"] / df["shifted_close"]
        print("ripples ratio is {}".format(df["ripples_ratio"]))




