# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# if __name__ == '__main__':

# train/validation/test 要根据时间点分，否则会有leakage
pos_df = pd.read_csv("pump_sample_raw.csv")
pos_df = pos_df[pos_df["exchange"] == "binance"]
pos_df = pos_df[pos_df["pair"] == "BTC"]
pos_df = pos_df[pos_df["pre_1h_price"].notna()]

neg_df = pd.read_csv("neg_sample_raw.csv", na_values="None")
neg_df = neg_df[neg_df["pair"] == "BTC"]
neg_df = neg_df[neg_df["pre_1h_price"].notna()]

#其余的使用平均值填充
for column in ['pre_3d_market_cap_usd','pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc','pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index', 'pre_3d_reddit_index', 'pre_3d_alexa_index']:
    mean_val = pos_df[column].mean()
    pos_df[column].fillna(mean_val, inplace=True)

for column in ['pre_3d_market_cap_usd','pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc','pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index', 'pre_3d_reddit_index', 'pre_3d_alexa_index']:
    mean_val = neg_df[column].mean()
    neg_df[column].fillna(mean_val, inplace=True)

price_columns = []
return_columns = []
volume_columns = []

for column in pos_df.columns:
    if "price" in column:
        price_columns.append(column)
    if "return" in column:
        return_columns.append(column)
    if "volume" in column:
        volume_columns.append(column)

other_columns = ['pre_3d_alexa_index', 'pre_3d_market_cap_btc', 'pre_3d_market_cap_usd', 'pre_3d_reddit_index', 'pre_3d_twitter_index']

for column in price_columns:
    pos_df[column] = pos_df[column] * (10**5)
    try:
        neg_df[column] = neg_df[column] * (10**5)
    except:
        continue

for column in volume_columns + other_columns:
    pos_df[column] = np.log2(pos_df[column]+0.1)
    neg_df[column] = np.log2(neg_df[column]+0.1)

pos_df["label"] = 1
neg_df["label"] = 0

# sequence 化
for idx, row in pos_df.iterrows():
    feature = []
    for column in price_columns + return_columns + volume_columns:
        feature.append(row[column])

    feature_str = "".join(map(lambda x: str(x), feature))
    pos_df.loc[idx, "feature"] = feature_str

# pre_feature_columns = []
X_pos = pd.merge(left=pos_df[["channel_id", "timestamp_unix"]], right=pos_df[["channel_id", "coin", "feature", "timestamp_unix"]], how='left', on=["channel_id"], sort=False)
X_pos = X_pos[X_pos.timestamp_unix_x > X_pos.timestamp_unix_y]
X_pos = X_pos.rename(columns={"timestamp_unix_x": "timestamp_unix_target",
                              "timestamp_unix_y": "timestamp_unix_seq",
                              "feature": "feature_seq",
                              "coin": "coin_seq"})

# channel_id, timestamp_unix 进行拼接，只用feature

def udf(df):
    def takeFirst(elem):
        return elem[0]
    # output = []
    feature_seq = []
    coin_seq = []
    X = list(zip(df.timestamp_unix_seq, df.coin_seq, df.feature_seq))
    X.sort(key=takeFirst, reverse=True)
    length = 0
    for x in X:  # set max length to 100
        coin_seq.append(str(x[1]))
        feature_seq.append(str(x[2]))
        length += 1
        if length >= 50:
            break
    return np.array(
        [[df.iloc[0]["channel_id"], df.iloc[0]["timestamp_unix_target"], str(length), "\t".join(coin_seq), "\t".join(feature_seq)]])

X_pos_final = X_pos.groupby(["channel_id", "timestamp_unix_target"]).apply(udf)
pos_seq_feat = pd.DataFrame(np.concatenate(X_pos_final.values, axis=0),
                               columns=["channel_id", "timestamp_unix", "length", "coin_seq", "feature_seq"])

pos_seq_feat.channel_id = pos_seq_feat.channel_id.astype(int)
pos_seq_feat.timestamp_unix = pos_seq_feat.timestamp_unix.astype(int)

pos_sample_base = pd.merge(left=pos_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])
neg_sample_base = pd.merge(left=neg_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])

column_list = "label,channel_id,coin,timestamp,length,coin_seq,feature_seq,pre_1h_return,pre_1h_price,pre_1h_price_avg,pre_1h_volume,pre_1h_volume_avg,pre_1h_volume_sum,pre_1h_volume_tb,pre_1h_volume_quote,pre_1h_volume_quote_tb,pre_3h_return,pre_3h_price,pre_3h_price_avg,pre_3h_volume,pre_3h_volume_avg,pre_3h_volume_sum,pre_3h_volume_tb,pre_3h_volume_tb_avg,pre_3h_volume_tb_sum,pre_3h_volume_quote,pre_3h_volume_quote_avg,pre_3h_volume_quote_sum,pre_3h_volume_quote_tb,pre_3h_volume_quote_tb_avg,pre_3h_volume_quote_tb_sum,pre_6h_return,pre_6h_price,pre_6h_price_avg,pre_6h_volume,pre_6h_volume_avg,pre_6h_volume_sum,pre_6h_volume_tb,pre_6h_volume_tb_avg,pre_6h_volume_tb_sum,pre_6h_volume_quote,pre_6h_volume_quote_avg,pre_6h_volume_quote_sum,pre_6h_volume_quote_tb,pre_6h_volume_quote_tb_avg,pre_6h_volume_quote_tb_sum,pre_12h_return,pre_12h_price,pre_12h_price_avg,pre_12h_volume,pre_12h_volume_avg,pre_12h_volume_sum,pre_12h_volume_tb,pre_12h_volume_tb_avg,pre_12h_volume_tb_sum,pre_12h_volume_quote,pre_12h_volume_quote_avg,pre_12h_volume_quote_sum,pre_12h_volume_quote_tb,pre_12h_volume_quote_tb_avg,pre_12h_volume_quote_tb_sum,pre_24h_return,pre_24h_price,pre_24h_price_avg,pre_24h_volume,pre_24h_volume_avg,pre_24h_volume_sum,pre_24h_volume_tb,pre_24h_volume_tb_avg,pre_24h_volume_tb_sum,pre_24h_volume_quote,pre_24h_volume_quote_avg,pre_24h_volume_quote_sum,pre_24h_volume_quote_tb,pre_24h_volume_quote_tb_avg,pre_24h_volume_quote_tb_sum,pre_36h_return,pre_36h_price,pre_36h_price_avg,pre_36h_volume,pre_36h_volume_avg,pre_36h_volume_sum,pre_36h_volume_tb,pre_36h_volume_tb_avg,pre_36h_volume_tb_sum,pre_36h_volume_quote,pre_36h_volume_quote_avg,pre_36h_volume_quote_sum,pre_36h_volume_quote_tb,pre_36h_volume_quote_tb_avg,pre_36h_volume_quote_tb_sum,pre_48h_return,pre_48h_price,pre_48h_price_avg,pre_48h_volume,pre_48h_volume_avg,pre_48h_volume_sum,pre_48h_volume_tb,pre_48h_volume_tb_avg,pre_48h_volume_tb_sum,pre_48h_volume_quote,pre_48h_volume_quote_avg,pre_48h_volume_quote_sum,pre_48h_volume_quote_tb,pre_48h_volume_quote_tb_avg,pre_48h_volume_quote_tb_sum,pre_60h_return,pre_60h_price,pre_60h_price_avg,pre_60h_volume,pre_60h_volume_avg,pre_60h_volume_sum,pre_60h_volume_tb,pre_60h_volume_tb_avg,pre_60h_volume_tb_sum,pre_60h_volume_quote,pre_60h_volume_quote_avg,pre_60h_volume_quote_sum,pre_60h_volume_quote_tb,pre_60h_volume_quote_tb_avg,pre_60h_volume_quote_tb_sum,pre_72h_return,pre_72h_price,pre_72h_price_avg,pre_72h_volume,pre_72h_volume_avg,pre_72h_volume_sum,pre_72h_volume_tb,pre_72h_volume_tb_avg,pre_72h_volume_tb_sum,pre_72h_volume_quote,pre_72h_volume_quote_avg,pre_72h_volume_quote_sum,pre_72h_volume_quote_tb,pre_72h_volume_quote_tb_avg,pre_72h_volume_quote_tb_sum,pre_3d_market_cap_usd,pre_3d_market_cap_btc,pre_3d_price_usd,pre_3d_price_btc,pre_3d_volume_usd,pre_3d_volume_btc,pre_3d_twitter_index,pre_3d_reddit_index,pre_3d_alexa_index".split(",")

neg_sample_base = neg_sample_base[column_list]
pos_sample_base = pos_sample_base[column_list]

hybrid_sample = pd.concat([neg_sample_base, pos_sample_base], axis=0)
hybrid_sample = hybrid_sample.reset_index(drop=True)

# shuffle
hybrid_sample.loc[hybrid_sample.length.isna(), "length"] = 0
hybrid_sample.loc[hybrid_sample.coin_seq.isna(), "coin_seq"] = "0"
hybrid_sample.loc[hybrid_sample.feature_seq.isna(), "feature_seq"] = "0"

hybrid_sample = hybrid_sample.loc[np.random.permutation(len(hybrid_sample))]
hybrid_sample = hybrid_sample.reset_index(drop=True)

# store
hybrid_sample.to_csv("test_sample.csv", index=False, header=False)

# do not forget to mannually split the training and testing set by time

