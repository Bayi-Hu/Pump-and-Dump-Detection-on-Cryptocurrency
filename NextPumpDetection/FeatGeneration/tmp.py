#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
import pickle as pkl
import json
from datetime import *
import time
import json

with open("binanceSymbol2CoinId.json", "r") as f:
    symbol2coinId = json.load(f)

cg = CoinGeckoAPI()
coin_list = cg.get_coins_list()

symbol2id = {}
for c in coin_list:
    symbol2id[c["symbol"]] = c["id"]

for s in symbol2coinId.keys():
    symbol2id[s.lower()] = symbol2coinId[s]

df = pd.read_csv("../../Telegram/Labeled/pump_attack_new.txt", sep="\t")
df["timestamp"] = df.timestamp.apply(pd.to_datetime)
df["timestamp_unix"] = (df["timestamp"].astype(int) / (10 ** 6)).astype(int)

coin_date_to_timestamp = {}
for idx, row in df.iterrows():
    attack_date = row.timestamp.strftime("%Y%m%d")
    attack_time = row.timestamp
    try:
        coin_date_to_timestamp[row.coin + "_" + attack_date].append(attack_time)
    except:
        coin_date_to_timestamp[row.coin + "_" + attack_date] = [attack_time]

error_key = []
coin_date_to_statistics_pre3d = {}

for i in range(len(coin_date_to_timestamp.keys())):
    key = list(coin_date_to_timestamp.keys())[i]
    coin_symbol, date = key.split("_")
    if coin_symbol not in ["ARN", "YOYO", 'ATM', 'BQX', 'BTCST', 'DATA', 'DREP', 'EDO',
                           'EZ', 'FLM', 'GNT', 'GXS', 'INS', 'ONG', 'QSP', 'TCT']:
        continue
    try:
        coin_id = symbol2id[coin_symbol.lower()]
        d = datetime.strptime(date, "%Y%m%d") + timedelta(days=-3)
        H = cg.get_coin_history_by_id(coin_id, date=d.strftime("%d-%m-%Y"))
        coin_date_to_statistics_pre3d[key] = H
    except:
        error_key.append(key)
    print(str(i) + ":" + key)
    time.sleep(1.5)

print("pause")

str_json = json.dumps(coin_date_to_statistics_pre3d)

print("pause")

