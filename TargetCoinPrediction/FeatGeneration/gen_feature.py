# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# if __name__ == '__main__':

# train/validation/test 要根据时间点分，否则会有leakage
pos_df = pd.read_csv("feature/pump_sample_raw.csv")
pos_df = pos_df[pos_df["exchange"] == "binance"]
pos_df = pos_df[pos_df["pair"] == "BTC"]
pos_df = pos_df[pos_df["pre_1h_price"].notna()]

neg_df = pd.read_csv("feature/neg_sample_raw.csv", na_values="None")
neg_df = neg_df[neg_df["pair"] == "BTC"]
neg_df = neg_df[neg_df["pre_1h_price"].notna()]

pos_df["label"] = 1
neg_df["label"] = 0

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

# 因为sequence feature要相加，所以去除了这一部分
# for column in price_columns:
#     if column == "pre_3d_price_usd":
#         continue
#     pos_df[column] = pos_df[column] * (10**5)
#     try:
#         neg_df[column] = neg_df[column] * (10**5)
#     except:
#         continue
#
# for column in volume_columns + other_columns:
#     pos_df[column] = np.log2(pos_df[column]+0.1)
#     neg_df[column] = np.log2(neg_df[column]+0.1)

# # 加入normalization
# pos_df
# neg_df
# # normalization
# hybrid_sample = pd.concat([train_sample, test_sample], axis=0)
# for column in column_list[7:]:
#     if column in column_list[-3:]:
#         pass
#     mean = hybrid_sample[column].mean()
#     std = hybrid_sample[column].std()
#     train_sample[column] = (train_sample[column] - mean) / std
#     test_sample[column] = (test_sample[column] - mean) / std

column1 = []
column2 = []

for column in price_columns + return_columns + volume_columns + other_columns:

    if column not in neg_df.columns:
        # after pump features
        mean = pos_df[column].mean()
        std = pos_df[column].std()
        pos_df[column] = (pos_df[column] - mean)/std

        column1.append(column)
    else:
        df = pd.concat([pos_df[column], neg_df[column]],axis=0)
        mean = df.mean()
        std = df.std()
        pos_df[column] = (pos_df[column] - mean) / std
        neg_df[column] = (neg_df[column] - mean) / std

        column2.append(column)


coin_df = pd.concat([pos_df["coin"], neg_df["coin"]], axis=0)
channel_df = pd.concat([pos_df["channel_id"], neg_df["channel_id"]], axis=0)

# channel_id & coin_id mapping
coin_array = np.unique(coin_df.values)
channel_array = np.unique(channel_df.values)
coin2id = {}
id2coin = {}
channel2id = {}
coin2id["0"] = 0
id2coin[0] = "0"
for i in range(len(coin_array)):
    coin2id[coin_array[i]] = i+1
    id2coin[i+1] = coin_array[i]

id2channel = {}
channel2id["0"] = 0
id2channel[0] = "0"

for i in range(len(channel_array)):
    channel2id[channel_array[i]] = i+1
    id2channel[i+1] = channel_array[i]

pos_df["coin_id"] = pos_df["coin"].apply(lambda x: coin2id[x])
pos_df["channel_id_new"] = pos_df["channel_id"].apply(lambda x: channel2id[x])

neg_df["coin_id"] = neg_df["coin"].apply(lambda x: coin2id[x])
neg_df["channel_id_new"] = neg_df["channel_id"].apply(lambda x: channel2id[x])

# for i, row in pos_df.iterrows():
#     pos_df.loc[i, "coin_id"] = coin2id[row.coin]
#     pos_df.loc[i, "channel_id_new"] = channel2id[row.channel_id]

# for i, row in neg_df.iterrows():
#     neg_df.loc[i, "coin_id"] = coin2id[row.coin]
#     neg_df.loc[i, "channel_id_new"] = channel2id[row.channel_id]

# sequence 化 !!!
for idx, row in pos_df.iterrows():
    feature = []
    for column in price_columns + return_columns + volume_columns + other_columns:
        feature.append(row[column])

    feature_str = "".join(map(lambda x: str(x), feature))
    pos_df.loc[idx, "feature"] = feature_str

# pre_feature_columns = []
X_pos = pd.merge(left=pos_df[["channel_id", "timestamp_unix"]], right=pos_df[["channel_id", "coin_id", "feature", "timestamp_unix"]], how='left', on=["channel_id"], sort=False)
X_pos = X_pos[X_pos.timestamp_unix_x > X_pos.timestamp_unix_y]
X_pos = X_pos.rename(columns={"timestamp_unix_x": "timestamp_unix_target",
                              "timestamp_unix_y": "timestamp_unix_seq",
                              "feature": "feature_seq",
                              "coin_id": "coin_id_seq"})

# channel_id, timestamp_unix 进行拼接，只用feature

def udf(df):
    def takeFirst(elem):
        return elem[0]
    # output = []
    feature_seq = []
    coin_id_seq = []
    X = list(zip(df.timestamp_unix_seq, df.coin_id_seq, df.feature_seq))
    X.sort(key=takeFirst, reverse=True)
    length = 0
    for x in X:  # set max length to 100
        coin_id_seq.append(str(x[1]))
        feature_seq.append(str(x[2]))
        length += 1
        if length >= 50:
            break
    return np.array(
        [[df.iloc[0]["channel_id"], df.iloc[0]["timestamp_unix_target"], str(length), "\t".join(coin_id_seq), "\t".join(feature_seq)]])

X_pos_final = X_pos.groupby(["channel_id", "timestamp_unix_target"]).apply(udf)
pos_seq_feat = pd.DataFrame(np.concatenate(X_pos_final.values, axis=0),
                               columns=["channel_id", "timestamp_unix", "length", "coin_id_seq", "feature_seq"])

pos_seq_feat.channel_id = pos_seq_feat.channel_id.astype(int)
pos_seq_feat.timestamp_unix = pos_seq_feat.timestamp_unix.astype(int)

pos_sample_base = pd.merge(left=pos_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])
neg_sample_base = pd.merge(left=neg_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])

column_list = "label,channel_id,channel_id_new,coin,coin_id,timestamp_unix,length,coin_id_seq,feature_seq,pre_1h_return,pre_1h_price,pre_1h_price_avg,pre_1h_volume,pre_1h_volume_avg,pre_1h_volume_sum,pre_1h_volume_tb,pre_1h_volume_quote,pre_1h_volume_quote_tb,pre_3h_return,pre_3h_price,pre_3h_price_avg,pre_3h_volume,pre_3h_volume_avg,pre_3h_volume_sum,pre_3h_volume_tb,pre_3h_volume_tb_avg,pre_3h_volume_tb_sum,pre_3h_volume_quote,pre_3h_volume_quote_avg,pre_3h_volume_quote_sum,pre_3h_volume_quote_tb,pre_3h_volume_quote_tb_avg,pre_3h_volume_quote_tb_sum,pre_6h_return,pre_6h_price,pre_6h_price_avg,pre_6h_volume,pre_6h_volume_avg,pre_6h_volume_sum,pre_6h_volume_tb,pre_6h_volume_tb_avg,pre_6h_volume_tb_sum,pre_6h_volume_quote,pre_6h_volume_quote_avg,pre_6h_volume_quote_sum,pre_6h_volume_quote_tb,pre_6h_volume_quote_tb_avg,pre_6h_volume_quote_tb_sum,pre_12h_return,pre_12h_price,pre_12h_price_avg,pre_12h_volume,pre_12h_volume_avg,pre_12h_volume_sum,pre_12h_volume_tb,pre_12h_volume_tb_avg,pre_12h_volume_tb_sum,pre_12h_volume_quote,pre_12h_volume_quote_avg,pre_12h_volume_quote_sum,pre_12h_volume_quote_tb,pre_12h_volume_quote_tb_avg,pre_12h_volume_quote_tb_sum,pre_24h_return,pre_24h_price,pre_24h_price_avg,pre_24h_volume,pre_24h_volume_avg,pre_24h_volume_sum,pre_24h_volume_tb,pre_24h_volume_tb_avg,pre_24h_volume_tb_sum,pre_24h_volume_quote,pre_24h_volume_quote_avg,pre_24h_volume_quote_sum,pre_24h_volume_quote_tb,pre_24h_volume_quote_tb_avg,pre_24h_volume_quote_tb_sum,pre_36h_return,pre_36h_price,pre_36h_price_avg,pre_36h_volume,pre_36h_volume_avg,pre_36h_volume_sum,pre_36h_volume_tb,pre_36h_volume_tb_avg,pre_36h_volume_tb_sum,pre_36h_volume_quote,pre_36h_volume_quote_avg,pre_36h_volume_quote_sum,pre_36h_volume_quote_tb,pre_36h_volume_quote_tb_avg,pre_36h_volume_quote_tb_sum,pre_48h_return,pre_48h_price,pre_48h_price_avg,pre_48h_volume,pre_48h_volume_avg,pre_48h_volume_sum,pre_48h_volume_tb,pre_48h_volume_tb_avg,pre_48h_volume_tb_sum,pre_48h_volume_quote,pre_48h_volume_quote_avg,pre_48h_volume_quote_sum,pre_48h_volume_quote_tb,pre_48h_volume_quote_tb_avg,pre_48h_volume_quote_tb_sum,pre_60h_return,pre_60h_price,pre_60h_price_avg,pre_60h_volume,pre_60h_volume_avg,pre_60h_volume_sum,pre_60h_volume_tb,pre_60h_volume_tb_avg,pre_60h_volume_tb_sum,pre_60h_volume_quote,pre_60h_volume_quote_avg,pre_60h_volume_quote_sum,pre_60h_volume_quote_tb,pre_60h_volume_quote_tb_avg,pre_60h_volume_quote_tb_sum,pre_72h_return,pre_72h_price,pre_72h_price_avg,pre_72h_volume,pre_72h_volume_avg,pre_72h_volume_sum,pre_72h_volume_tb,pre_72h_volume_tb_avg,pre_72h_volume_tb_sum,pre_72h_volume_quote,pre_72h_volume_quote_avg,pre_72h_volume_quote_sum,pre_72h_volume_quote_tb,pre_72h_volume_quote_tb_avg,pre_72h_volume_quote_tb_sum,pre_3d_market_cap_usd,pre_3d_market_cap_btc,pre_3d_price_usd,pre_3d_price_btc,pre_3d_volume_usd,pre_3d_volume_btc,pre_3d_twitter_index,pre_3d_reddit_index,pre_3d_alexa_index".split(",")

pos_sample_base = pos_sample_base[column_list]
neg_sample_base = neg_sample_base[column_list]

pos_sample_base.loc[pos_sample_base.length.isna(), "length"] = 0
pos_sample_base.loc[pos_sample_base.coin_id_seq.isna(), "coin_id_seq"] = "0"
pos_sample_base.loc[pos_sample_base.feature_seq.isna(), "feature_seq"] = "0"

neg_sample_base.loc[neg_sample_base.length.isna(), "length"] = 0
neg_sample_base.loc[neg_sample_base.coin_id_seq.isna(), "coin_id_seq"] = "0"
neg_sample_base.loc[neg_sample_base.feature_seq.isna(), "feature_seq"] = "0"

# split
split_timestamp_unix = 1620579621000

test_pos_sample_base = pos_sample_base[pos_sample_base.timestamp_unix >= split_timestamp_unix]
train_pos_sample_base = pos_sample_base[pos_sample_base.timestamp_unix < split_timestamp_unix]

test_neg_sample_base = neg_sample_base[neg_sample_base.timestamp_unix >= split_timestamp_unix]
train_neg_sample_base = neg_sample_base[neg_sample_base.timestamp_unix < split_timestamp_unix]

train_sample = pd.concat([train_pos_sample_base, train_neg_sample_base], axis=0)
test_sample = pd.concat([test_pos_sample_base, test_neg_sample_base], axis=0)

train_sample = train_sample.reset_index(drop=True)
test_sample = test_sample.reset_index(drop=True)

# shuffle
train_sample = train_sample.loc[np.random.permutation(len(train_sample))]
train_sample = train_sample.reset_index(drop=True)

test_sample = test_sample.loc[np.random.permutation(len(test_sample))]
test_sample = test_sample.reset_index(drop=True)



train_sample = train_sample[column_list]
test_sample = test_sample[column_list]

train_sample.coin_id = train_sample.coin_id.astype(int)
train_sample.channel_id_new = train_sample.channel_id_new.astype(int)

test_sample.coin_id = test_sample.coin_id.astype(int)
test_sample.channel_id_new = test_sample.channel_id_new.astype(int)

# store
train_sample.to_csv("train_sample.csv", index=False, header=False)
test_sample.to_csv("test_sample.csv", index=False, header=False)
# do not forget to mannually split the training and testing set by time

