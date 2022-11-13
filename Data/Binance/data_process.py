import zipfile
import os
import pandas as pd
from datetime import *

def unzip(src_file, dest_dir):
    """ungz zip file"""
    zf = zipfile.ZipFile(src_file)
    zf.extractall(path=dest_dir)
    zf.close()

if __name__ == '__main__':

    dest_dir = "./data/unzip"
    file_list = []
    for root, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith(".zip"):

                unzip(os.path.join(root, file), dest_dir)

    print("pause")
    dest_dir = "./data/concat"

    # concate the last month data with current month data
    df = pd.read_csv("Telegram/Labeled/PD_logs_cleaned.txt", sep="\t")
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)

    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
               "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

    for i, row in df.iterrows():

        if row.timestamp.day > 3:
            last_month_timestamp = row.timestamp - timedelta(days=31)

        else:
            last_month_timestamp = row.timestamp - timedelta(days=15)

        try:
            current_month_file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            current_month_statistics = pd.read_csv("./data/unzip/" + current_month_file_name, names=columns)
            last_month_file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + last_month_timestamp.strftime("%Y-%m") + ".csv"
            last_month_statistics = pd.read_csv("./data/unzip/" + last_month_file_name, names=columns)

            current_month_statistics = pd.concat([last_month_statistics, current_month_statistics], axis=0)
            current_month_statistics.to_csv(os.path.join(dest_dir, current_month_file_name), index=False)

        except:
            continue





