import pandas as pd
from unittest import TestCase
import matplotlib.pyplot as plt

# df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]


class TestMACD(TestCase):
    """
    calculate the MACD value
    12
    """
    def test_cal_macd(self,df,fastperiod=12,slowperiod=26,signalperiod=9):
        ewma12 = df["Close"].ewm(span=fastperiod,adjust=False).mean()  # adjust设置为False非常关键
        ewma26 = df["Close"].ewm(span=slowperiod,adjust=False).mean()
        df["dif"] = ewma12 - ewma26
        df["dea"] = df["dif"].ewm(span=signalperiod,adjust=False).mean()
        df["bar"] = (df["dif"] - df["dea"])*2
        return df

    def test_macd(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df_macd = self.test_cal_macd(df)
        print(df_macd)

        plt.figure()
        df_macd['dea'].plot(color='red', label='dea')
        df_macd['dif'].plot(color='blue', label='dif')
        plt.legend(loc='best')

        pos_bar = []
        pos_index = []
        neg_bar = []
        neg_index = []

        for index,row in df.iterrows():









