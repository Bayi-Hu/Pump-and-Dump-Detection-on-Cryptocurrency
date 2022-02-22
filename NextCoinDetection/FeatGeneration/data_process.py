#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# train/validation/test 要根据时间点分，否则会有leakage
df = pd.read_csv("../../Telegram/Labeled/pump_attack_new.txt", sep="\t")
df["timestamp"] = df.timestamp.apply(pd.to_datetime)
df["timestamp_unix"] = (df["timestamp"].astype(int) / (10**6)).astype(int)


columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
          "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]


# after_pump_feature
df["insider_return"] = -1.0
df["outsider_return"] = -1.0
df["outsider_avg_return"] = -1.0
df["outsider_next_avg_return"] = -1.0


def pre_pump_statistics(statistics, idx, max_hour=72):
    prices = []
    volumes = []
    volumes_taker_buy_base = []

    for j in range(max_hour + 1):
        #     for j in range(200):
        C = 0
        V = 0
        V_taker_buy_base = 0

        for i in range(idx - (j + 1) * 60, idx - j * 60):
            p = float(statistics.loc[i, "high"] + statistics.loc[i, "low"]) / 2
            v = float(statistics.loc[i, "volume"])
            v_taker_buy = float(statistics.loc[i, "taker_buy_base_asset_volume"])

            C += p * v
            V += v
            V_taker_buy_base += v_taker_buy

        if V != 0:
            AggregatedPrice = C / V
            prices.append(AggregatedPrice)

        volumes.append(V)
        volumes_taker_buy_base.append(V_taker_buy_base)

    return prices, volumes

if __name__ == '__main__':

    debug_cnt1 = 0
    debug_cnt2 = 0
    for i in range(len(df)):
        if df.loc[i, "exchange"] != "binance":
            continue
        try:
            file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            statistics = pd.read_csv("../../CoinStatistics/data/unzip/" + file_name, names=columns)
            debug_cnt1 += 1

        except:
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
            prices, volumes = pre_pump_statistics(statistics, idx, max_hour=96)

            return_rate = []
            W = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96]
            for w in W:
                return_rate.append(prices[0] / prices[w] - 1.0)
                df.loc[i, "pre" + str(w) + "h_return"] = prices[0] / prices[w] - 1.0

            return_rate_list.append(return_rate)
            debug_cnt2 += 1

        except:
            continue

    print("pause")
# pump前的特征
# pump后的特征




