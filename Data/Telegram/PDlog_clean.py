import pandas as pd
import numpy as np

df = pd.read_csv("Labeled/PD_logs_raw.txt", header=None, sep="\t", names=["channel_id", "session_id", "coin", "exchange", "pair", "timestamp"])
df["timestamp"] = df.timestamp.apply(pd.to_datetime)
coin_time_to_exchange = {}
coin_time_to_pair = {}

for idx, row in df[df["pair"] != "?"].iterrows():
    coin_time_to_pair[row.coin + "_" + row.timestamp.strftime("%Y%m%d")] = row.pair

for idx, row in df[df["exchange"] != "?"].iterrows():
    coin_time_to_exchange[row.coin + "_" + row.timestamp.strftime("%Y%m%d")] = row.exchange

exchange_fail_cnt = 0
pair_fail_cnt = 0

for i, r in df.iterrows():
    if df.loc[i, "exchange"] == "?":
        try:
            exchange = coin_time_to_exchange[df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")]
            df.loc[i, "exchange"] = exchange

        except:
            exchange_fail_cnt += 1

    if df.loc[i, "pair"] == "?":
        try:
            pair = coin_time_to_pair[df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")]
            df.loc[i, "pair"] = pair

        except:
            pair_fail_cnt += 1


for i in range(len(df)):
    if df.loc[i, "exchange"] == "?":
        if (df.loc[i - 1, "exchange"] != "?" and df.loc[i - 1, "channel_id"] == df.loc[i, "channel_id"]):
            df.loc[i, "exchange"] = df.loc[i - 1, "exchange"]

    if df.loc[i, "pair"] == "?":
        if (df.loc[i - 1, "pair"] != "?" and df.loc[i - 1, "channel_id"] == df.loc[i, "channel_id"]):
            df.loc[i, "pair"] = df.loc[i - 1, "pair"]

for i in range(len(df)):

    if df.loc[i, "pair"] == "?":
        df.loc[i, "pair"] = "BTC"

    if df.loc[i, "exchange"] == "?":
        df.loc[i, "exchange"] = "binance"

df.to_csv("./Labeled/PD_logs_cleaned.txt", index=False, sep="\t")