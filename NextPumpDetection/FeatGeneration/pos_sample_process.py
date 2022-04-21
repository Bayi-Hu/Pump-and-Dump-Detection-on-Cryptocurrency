#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
import pickle as pkl
import json
from datetime import *
import time

with open("raw/binanceSymbol2CoinId.json", "r") as f:
    symbol2coinId = json.load(f)

cg = CoinGeckoAPI()
coin_list = cg.get_coins_list()

symbol2id = {}
for c in coin_list:
    symbol2id[c["symbol"]] = c["id"]

for s in symbol2coinId.keys():
    symbol2id[s.lower()] = symbol2coinId[s]


def pre_pump_statistics(statistics, idx, bucket_num=72, bucket_size_min=60):
    price_list = []
    volume_list = []
    volume_tb_list = []
    volume_q_list = []
    volume_tb_q_list = []

    for j in range(bucket_num + 1):
        C = 0
        V = 0
        V_tb = 0
        V_q = 0
        V_q_tb = 0
        prices = []

        for i in range(idx - (j + 1) * bucket_size_min, idx - j * bucket_size_min):
            p = float(statistics.loc[i, "high"] + statistics.loc[i, "low"]) / 2
            v = float(statistics.loc[i, "volume"])
            v_tb = float(statistics.loc[i, "taker_buy_base_asset_volume"])
            v_q = float(statistics.loc[i, "quote_asset_volume"])
            v_q_tb = float(statistics.loc[i, "taker_buy_quote_asset_volume"])

            C += p * v
            V += v
            V_tb += v_tb
            V_q += v_q
            V_q_tb += v_q_tb
            prices.append(p)

        if V != 0:
            AggregatedPrice = C / V

        else:
            AggregatedPrice = np.mean(prices)

        price_list.append(AggregatedPrice)
        volume_list.append(V)
        volume_tb_list.append(V_tb)
        volume_q_list.append(V_q)
        volume_tb_q_list.append(V_q_tb)

    return price_list, volume_list, volume_tb_list, volume_q_list, volume_tb_q_list

if __name__ == '__main__':

    debug_cnt1 = 0
    debug_cnt2 = 0
    debug_idx1 = []
    debug_idx2 = []

    # train/validation/test 要根据时间点分，否则会有leakage
    df = pd.read_csv("../../Telegram/Labeled/pump_attack_new.txt", sep="\t")
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)
    df["timestamp_unix"] = (df["timestamp"].astype(int) / (10 ** 6)).astype(int)

    for i in range(len(df)):

        if df.loc[i, "exchange"] != "binance":
            continue
        try:
            file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            statistics = pd.read_csv("../../CoinStatistics/data/concat/" + file_name)
            debug_cnt1 += 1

        except:
            debug_idx1.append(i)
            continue

        statistics["open_scale"] = statistics["open"] * 10 ** 8
        statistics["close_scale"] = statistics["close"] * 10 ** 8
        statistics["high_scale"] = statistics["high"] * 10 ** 8
        statistics["low_scale"] = statistics["low"] * 10 ** 8
        statistics["maker_buy_base_asset_volume"] = statistics["volume"] - statistics["taker_buy_base_asset_volume"]
        statistics["maker_buy_quote_asset_volume"] = statistics["quote_asset_volume"] - statistics["taker_buy_quote_asset_volume"]

        idx = np.max(statistics[statistics.open_time < df.loc[i, "timestamp_unix"]].index)

        insider_price = (statistics.loc[idx-60, "high"] + statistics.loc[idx-60, "low"]) / 2
        outsider_price = statistics.loc[idx, "open"]
        highest_price = np.max(statistics.loc[range(idx, idx + 60), "high"].values)
        outsider_avg_price = (statistics.loc[idx, "high"] + statistics.loc[idx, "low"]) / 2
        outsider_next_avg_price = (statistics.loc[idx + 1, "high"] + statistics.loc[idx + 1, "low"]) / 2

        insider_return = highest_price / insider_price
        outsider_return = highest_price / outsider_price
        outsider_avg_return = highest_price / outsider_avg_price
        outsider_next_avg_return = outsider_next_avg_price / outsider_avg_price

        # after pump
        df.loc[i, "insider_price"] = insider_price
        df.loc[i, "outsider_price"] = outsider_price
        df.loc[i, "highest_price"] = highest_price
        df.loc[i, "outsider_avg_price"] = outsider_avg_price
        df.loc[i, "outsider_next_avg_price"] = outsider_next_avg_price

        df.loc[i, "insider_return"] = insider_return
        df.loc[i, "outsider_return"] = outsider_return
        df.loc[i, "outsider_avg_return"] = outsider_avg_return
        df.loc[i, "outsider_next_avg_return"] = outsider_next_avg_return

        # before pump
        idx = np.max(statistics[statistics.open_time < df.loc[i, "timestamp_unix"]].index)
        idx = idx - 30

        try:

            pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list, pre_volume_tb_q_list = pre_pump_statistics(statistics, idx, bucket_num=72, bucket_size_min=60)
            return_rate = []
            W = [1, 3, 6, 12, 24, 36, 48, 60, 72]
            for w in W:

                df.loc[i, "pre_" + str(w) + "h_return"] = pre_price_list[0] / pre_price_list[w] - 1.0
                df.loc[i, "pre_"+ str(w) + "h_price"] = pre_price_list[w-1]
                df.loc[i, "pre_"+ str(w) + "h_price_avg"] = np.mean(pre_price_list[:w])
                df.loc[i, "pre_"+ str(w) + "h_volume"] = np.sum(pre_volume_list[w-1])
                df.loc[i, "pre_" + str(w) + "h_volume_avg"] = np.mean(pre_volume_list[:w])
                df.loc[i, "pre_" + str(w) + "h_volume_sum"] = np.sum(pre_volume_list[:w])

                df.loc[i, "pre_" + str(w) + "h_volume_tb"] = pre_volume_tb_list[w - 1]
                if w > 1:
                    df.loc[i, "pre_" + str(w) + "h_volume_tb_avg"] = np.mean(pre_volume_tb_list[:w])
                    df.loc[i, "pre_" + str(w) + "h_volume_tb_sum"] = np.sum(pre_volume_tb_list[:w])

                df.loc[i, "pre_" + str(w) + "h_volume_quote"] = pre_volume_q_list[w - 1]
                if w > 1:
                    df.loc[i, "pre_" + str(w) + "h_volume_quote_avg"] = np.mean(pre_volume_q_list[:w])
                    df.loc[i, "pre_" + str(w) + "h_volume_quote_sum"] = np.sum(pre_volume_q_list[:w])

                df.loc[i, "pre_" + str(w) + "h_volume_quote_tb"] = pre_volume_tb_q_list[w - 1]

                if w > 1:
                    df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_avg"] = np.mean(pre_volume_tb_q_list[:w])
                    df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_sum"] = np.sum(pre_volume_tb_q_list[:w])

            debug_cnt2 += 1

        except:
            debug_idx2.append(i)
            continue

    with open("raw/coinDate2Statistics_pred3d.json", "r") as f:
        coin_date_to_statistics_pre3d = json.load(f)


    debug_cnt3 = 0
    debug_idx3 = []
    for i in range(len(df)):

        key = df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")
        try:
            market_cap_usd = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["usd"]
            market_cap_btc = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["btc"]
            price_usd = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["usd"]
            price_btc = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["btc"]
            volume_usd = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["usd"]
            volume_btc = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["btc"]

            twitter_follower = coin_date_to_statistics_pre3d[key]["community_data"]["twitter_followers"]
            reddit_subscriber = coin_date_to_statistics_pre3d[key]["community_data"]["reddit_subscribers"]
            alexa_rank = coin_date_to_statistics_pre3d[key]["public_interest_stats"]["alexa_rank"]

            if market_cap_usd != 0:
                df.loc[i, "pre_3d_market_cap_usd"] = market_cap_usd

            if market_cap_btc != 0:
                df.loc[i, "pre_3d_market_cap_btc"] = market_cap_btc

            df.loc[i, "pre_3d_price_usd"] = price_usd
            df.loc[i, "pre_3d_price_btc"] = price_btc

            df.loc[i, "pre_3d_volume_usd"] = volume_usd
            df.loc[i, "pre_3d_volume_btc"] = volume_btc

            if twitter_follower:
                df.loc[i, "pre_3d_twitter_index"] = twitter_follower

            if reddit_subscriber:
                df.loc[i, "pre_3d_reddit_index"] = reddit_subscriber

            if alexa_rank:
                df.loc[i, "pre_3d_alexa_index"] = alexa_rank

            debug_cnt3 += 1
        except:
            debug_idx3.append(key)
            continue

    print("pause")
    df.to_csv("pump_sample_raw.csv", index = False)







