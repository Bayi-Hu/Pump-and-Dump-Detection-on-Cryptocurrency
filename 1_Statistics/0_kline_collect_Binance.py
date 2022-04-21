"""
  script to download klines.
  set the absoluate path destination folder for STORE_DIRECTORY, and run

  e.g. STORE_DIRECTORY=/data/ ./download-kline.py

"""
import sys
from datetime import *
import pandas as pd
from enums import *
from utility import download_monthly_klines, download_daily_klines

if __name__ == "__main__":

    df = pd.read_csv("../0_TelegramData/Labeled/pump_attack_new.txt", sep="\t")
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)

    for i, row in df.iterrows():
        print(row)
        try:
            if row.timestamp.day > 3:
                last_month_timestamp = row.timestamp - timedelta(days=31)

            else:
                last_month_timestamp = row.timestamp - timedelta(days=15)

            download_monthly_klines(trading_type="spot",
                                    symbols=[row.coin + row.pair],
                                    num_symbols=1,
                                    intervals=["1m"],
                                    years=[last_month_timestamp.year],
                                    months=[last_month_timestamp.month],
                                    start_date=None,
                                    end_date=None,
                                    folder="",
                                    checksum=0
                                    )

        except:
            continue
