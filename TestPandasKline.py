import pandas as pd
from unittest import TestCase
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ochl
import mplfinance as mpf


class TestPandasKline(TestCase):
    def test_kline_chart(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]

        fig = plt.figure()
        axes = fig.add_subplot(111)
        candlestick2_ochl(
            ax=axes,
            opens=df["Open"].values,
            highs=df["High"].values,
            lows=df["Low"].values,
            closes=df["Close"].values,
            width=0.75,
            colorup='red',
            colordown='green'
        )
        plt.xticks(range(len(df.index.values)),df.index.values,rotation=30)
        axes.grid(True)
        plt.title("Kline Chart")
        plt.show()

    def test_kline_volume(self):
        file_name = "stock_data.csv"
        df = pd.read_csv(file_name)
        df.columns = ["Code", "Date", "Close", "Open", "High", "Low", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        print(df)

        my_color = mpf.make_marketcolors(
            up='red',
            down='green',
            volume={'up':'red','down':'green'}
        )
        my_style = mpf.make_mpf_style(
            marketcolors=my_color,
            gridaxis='both',
            gridstyle='-.',
            rc={'font.family': 'STSong'}

        )
        mpf.plot(
            df,
            type= 'candle',
            title='Kline Chart by volume',
            ylabel='price',
            show_nontrading=False,
            style=my_style,
            volume=True,
            ylabel_lower='volume',
            datetime_format="%Y-%m-%d",
            xrotation=45,
            tight_layout=False
        )








