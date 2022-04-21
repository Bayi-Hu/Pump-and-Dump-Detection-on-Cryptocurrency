#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
import pickle as pkl
from datetime import *
from CoinStatistics.utility import download_monthly_klines
import zipfile
import os
import time
from pos_sample_process import pre_pump_statistics

# markets_coin_list = []
# for i in range(4000):
#     markets = cg.get_coins_markets(vs_currency="usd", order = "market_cap_desc", per_page=100, page=i)
#     markets_coin_list += markets

binance_symbol = "HNT,BOND,WOO,ZEN,GRS,AAVE,LINKDOWN,DCR,UMA,FORTH,RVN,CVX,STEEM,RNDR,KSM,BAR,XTZUP,FTT,ERD,CKB,FXS,MCO,SUB,JUV,BNX,GHST,WING,CHESS,LTC,TRXDOWN,CHAT,XRPBULL,BCHUP,SNM,DGB,STPT,EGLD,TWT,SAND,XMR,BICO,TROY,REN,BAND,ACH,MBOX,GLMR,IOST,DAI,CDT,ATM,SOL,USDC,ARN,BRD,ETHBULL,FILDOWN,RSR,ADA,BURGER,YGG,TKO,SKY,BNBBEAR,OGN,CLOAK,FRONT,PERL,CLV,LOKA,RAY,WAN,EASY,AST,COTI,LINK,PAXG,GVT,KNC,DENT,LUN,NPXS,DOGE,OOKI,HIVE,XLMDOWN,MBL,GXS,FIS,NULS,TLM,ENJ,STRAT,PERP,PUNDIX,C98,FLOW,NAV,XTZDOWN,NAS,RDN,XEC,DYDX,ZRX,VEN,NEO,SRM,ANT,TORN,QNT,ICN,USDP,EOS,UTK,IDEX,GAS,BTT,REEF,ETH,LTO,JST,ALPHA,LRC,EOSUP,RCN,ARPA,ATA,BTTC,HOT,MATIC,FIO,THETA,MDX,DOT,COMP,KEEP,ENG,ASR,UNIDOWN,GALA,ELF,STMX,XRPUP,LINKUP,SNX,SANTOS,DF,TNB,KEY,VIBE,BUSD,DASH,RLC,SXP,NXS,POA,SFP,SPELL,ETHBEAR,BCHDOWN,BAT,BAKE,OM,ROSE,TRXUP,CELO,XVS,KP3R,ETHDOWN,VTHO,MDA,VET,BCHSV,ICP,NU,ACA,AUTO,AION,SNGLS,UST,FARM,BULL,QKC,HARD,AUD,USDSB,BCH,EOSBEAR,WRX,TRX,FIL,NBS,EZ,DODO,OST,REP,AMP,LUNA,LTCDOWN,FOR,CAKE,VITE,MKR,WIN,LOOM,STRAX,SUPER,MASK,LIT,EOSDOWN,MITH,TCT,BEAM,AKRO,ADX,LSK,MANA,GNO,FUN,WAVES,MOVR,SALT,TRB,POLS,UNIUP,ICX,ATOM,GTO,TVK,CHR,VOXEL,WBTC,YOYO,STORJ,EUR,SKL,APPC,AGI,ALPACA,RENBTC,SNT,GLM,MINA,AUDIO,STX,CTK,MTL,GO,NMR,LEND,PYR,BADGER,QUICK,PHX,NKN,GBP,ALPINE,FIRO,BAL,QSP,SUN,VIDT,ERN,FLM,XVG,STORM,DOTUP,AAVEDOWN,ONT,ARDR,RARE,REQ,NEBL,ALGO,BOT,PROM,DEXE,QTUM,HBAR,DOCK,CFX,ALCX,SHIB,ANY,SYS,WTC,CTXC,KLAY,OXT,CELR,BTCUP,DOTDOWN,XEM,SSV,BETA,CVP,1INCH,EOSBULL,FIDA,SUSHI,SXPDOWN,FUEL,ADADOWN,RIF,ENS,WAXP,DEGO,IOTX,BCD,PEOPLE,VGX,ANKR,UNFI,BTC,DNT,SUSHIDOWN,API3,HIGH,1INCHUP,HSR,DIA,OG,1INCHDOWN,AVA,BTCB,INS,AMB,SUSD,FTM,POND,OCEAN,BCPT,SUSHIUP,PNT,ANC,NANO,ONE,POWR,TOMO,BNT,DAR,YFII,IMX,IRIS,BEL,GRT,SC,ZIL,PLA,OAX,YFI,LPT,MLN,EDO,WINGS,AE,XNO,MFT,XLM,PPT,ACM,BTCDOWN,SLP,BTCST,HC,XTZ,AGLD,LTCUP,LAZIO,RAD,BCC,SCRT,PHA,BNB,FLUX,RUNE,EVX,BTG,DREP,XRPBEAR,TRIG,XRPDOWN,FILUP,WPR,PHB,OMG,GTC,DUSK,NEAR,IOTA,RAMP,SXPUP,PIVX,COS,INJ,BKRW,TNT,XRP,AVAX,EPS,MOD,QI,CND,QLC,KMD,TRU,GNT,JOE,RPX,ALICE,COCOS,NCASH,BLZ,USDS,XLMUP,CRV,TRIBE,BNBDOWN,ZEC,BTS,POE,XZC,CTSI,ILV,RGT,AERGO,AUCTION,UNI,LINA,ARK,FET,CHZ,POLY,AAVEUP,AGIX,YFIDOWN,BQX,MC,ETC,MTH,WNXM,BNBUP,BEAR,BCHABC,BNBBULL,ETHUP,TUSD,CVC,ADAUP,BCN,MIR,VIA,CITY,PORTO,KAVA,AXS,DLT,CMT,AR,WABI,PSG,TFUEL,VIB,JASMY,ORN,DGD,PAX,ONG,DATA,YFIUP,MDT,BZRX"
binance_symbol_list = binance_symbol.split(",")
require_binance_file_names = set()
require_coingecko_file_names = set()

download_flag = False
zip_flag = False
concat_flag = False
coin_gecko_flag = False
gecko_statistics = False


if __name__ == '__main__':
    
    # train/validation/test 要根据时间点分，否则会有leakage
    df = pd.read_csv("../../TelegramData/Labeled/pump_attack_new.txt", sep="\t")
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)
    df["timestamp_unix"] = (df["timestamp"].astype(int) / (10 ** 6)).astype(int)

    try:
        require_binance_file_names = pd.read_csv("raw/neg_binance_required_file_name.txt", names=["file_name"]).file_name.values
    except:
        for i in range(len(df)):
            if df.loc[i, "exchange"] != "binance":
                continue
            for symbol in binance_symbol_list:
                if symbol == df.loc[i, "coin"]:
                    continue
                file_name = symbol + df.loc[i, "pair"] + "-" + df.loc[i, "timestamp"].strftime("%Y%m")
                require_binance_file_names.add(file_name)

        with open("raw/neg_binance_required_file_name.txt", "w") as f:
            for file_name in require_binance_file_names:
                f.write(file_name+"\n")

    if download_flag:
        last_month_file_names = set()
        for i in range(0, len(require_binance_file_names)):
            file_name = require_binance_file_names[i]
            symbol = file_name.split("-")[0]
            date = datetime.strptime(file_name.split("-")[1], "%Y%m")
            download_monthly_klines(trading_type="spot",
                                    symbols=[symbol],
                                    num_symbols=1,
                                    intervals=["1m"],
                                    years=[date.year],
                                    months=[date.month],
                                    start_date=None,
                                    end_date=None,
                                    folder="",
                                    checksum=0
                                    )
            if date.day > 3:
                last_month_date = date - timedelta(days=31)
            else:
                last_month_date = date - timedelta(days=15)

            last_month_file_name = symbol+"-"+date.strftime("%Y%m")
            if last_month_file_name not in last_month_file_names and last_month_file_name not in require_binance_file_names:
                last_month_file_names.add(last_month_file_name)
                download_monthly_klines(trading_type="spot",
                                        symbols=[symbol],
                                        num_symbols=1,
                                        intervals=["1m"],
                                        years=[last_month_date.year],
                                        months=[last_month_date.month],
                                        start_date=None,
                                        end_date=None,
                                        folder="",
                                        checksum=0
                                        )

    if zip_flag:
        def unzip(src_file, dest_dir):
            """ungz zip file"""
            zf = zipfile.ZipFile(src_file)
            zf.extractall(path=dest_dir)
            zf.close()

        dest_dir = "../../CoinStatistics/data/unzip"
        fail_file_list = []
        for root, dirs, files in os.walk("../../CoinStatistics/data/spot"):
            for file in files:
                if file.endswith(".zip"):
                    try:
                        unzip(os.path.join(root, file), dest_dir)
                    except:
                        fail_file_list.append(file)

    if concat_flag:

        dest_dir = "../../CoinStatistics/data/concat"
        columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
                   "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

        # concate the last month data with current month data
        fail_file_set = set()
        for file_name in require_binance_file_names:
            symbol = file_name.split("-")[0]
            date = datetime.strptime(file_name.split("-")[1], "%Y%m")
            if date.day > 3:
                last_month_date = date - timedelta(days=31)
            else:
                last_month_date = date - timedelta(days=15)
            try:
                current_month_file_name = symbol + "-1m-" + date.strftime("%Y-%m") + ".csv"
                current_month_statistics = pd.read_csv("../../CoinStatistics/data/unzip/" + current_month_file_name, names=columns)
                last_month_file_name = symbol + "-1m-" + last_month_date.strftime("%Y-%m") + ".csv"
                last_month_statistics = pd.read_csv("../../CoinStatistics/data/unzip/" + last_month_file_name, names=columns)
                current_month_statistics = pd.concat([last_month_statistics, current_month_statistics], axis=0)
                current_month_statistics.to_csv(os.path.join(dest_dir, current_month_file_name), index=False)
            except:
                fail_file_set.add(file_name)
                continue

    if gecko_statistics:

        import json
        with open("raw/binanceSymbol2CoinId.json", "r") as f:
            symbol2coinId = json.load(f)
        cg = CoinGeckoAPI()
        coin_list = cg.get_coins_list()
        symbol2id = {}
        for c in coin_list:
            symbol2id[c["symbol"]] = c["id"]
        for s in symbol2coinId.keys():
            symbol2id[s.lower()] = symbol2coinId[s]

        file_name_list = []
        for root, dirs, files in os.walk("../../CoinStatistics/data/concat"):
            for file in files:
                if file.endswith(".csv"):
                    file_name_list.append(file)

        unattack_coin_date = set()
        count = 0
        for i in range(len(df)):
            if df.loc[i, "exchange"] != "binance":
                continue
            for symbol in binance_symbol_list:
                if symbol == df.loc[i, "coin"]:
                    continue

                file_name = symbol + df.loc[i, "pair"] + "-1m-"  + df.loc[i, "timestamp"].strftime("%Y-%m")+".csv"
                if file_name in file_name_list:
                    pre_3d = df.loc[i, "timestamp"] + timedelta(days=-3)
                    key = symbol + "_" + pre_3d.strftime("%Y%m%d")
                    unattack_coin_date.add(key)
                    count += 1

        print("pause")
        debug_cnt1 = 0
        debug_cnt2 = 0
        debug_key1 = []
        debug_key2 = []
        neg_coin_date_to_statistics = {}
        f = open("raw/neg_coin_gecko_statistics.txt", "w")
        for key in unattack_coin_date:
            symbol = key.split("_")[0]
            date = key.split("_")[1]
            try:
                coin_id = symbol2coinId[symbol]
            except:
                debug_cnt1 += 1
                debug_key1.append(symbol)
                continue
            date = datetime.strptime(date, "%Y%m%d").strftime("%d-%m-%Y")
            try:
                H = cg.get_coin_history_by_id(coin_id, date=date)
                market_cap_usd = H["market_data"]["market_cap"]["usd"]
                market_cap_btc = H["market_data"]["market_cap"]["btc"]
                price_usd = H["market_data"]["current_price"]["usd"]
                price_btc = H["market_data"]["current_price"]["btc"]
                volume_usd = H["market_data"]["total_volume"]["usd"]
                volume_btc = H["market_data"]["total_volume"]["btc"]
                twitter_follower = H["community_data"]["twitter_followers"]
                reddit_subscriber = H["community_data"]["reddit_subscribers"]
                alexa_rank = H["public_interest_stats"]["alexa_rank"]

                statistic_data = [key, market_cap_usd, market_cap_btc, price_usd, price_btc, volume_usd, volume_btc, twitter_follower, reddit_subscriber, alexa_rank]
                f.write("\t".join(map(lambda x: str(x), statistic_data)) + "\n")

            except:
                debug_cnt2 += 1
                debug_key2.append(key)
                continue

            time.sleep(1.3)

    # generate neg_sample by combining binance price history and coingecko statistics

    file_name_list = []
    for root, dirs, files in os.walk("../../CoinStatistics/data/concat"):
        for file in files:
            if file.endswith(".csv"):
                file_name_list.append(file)

    count = 0

    # 先产生一个neg_sample_df

    neg_sample_df_symbol = []
    neg_sample_df_pair = []
    neg_sample_df_timestamp = []
    neg_sample_df_channel = []
    neg_sample_df_pre3d = []

    for i in range(len(df)):
        if df.loc[i, "exchange"] != "binance":
            continue
        for symbol in binance_symbol_list:
            if symbol == df.loc[i, "coin"]:
                continue
            file_name = symbol + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            if file_name in file_name_list:
                neg_sample_df_channel.append(df.loc[i,"channel_id"])
                neg_sample_df_symbol.append(symbol)
                neg_sample_df_pair.append(df.loc[i, "pair"])
                neg_sample_df_timestamp.append(df.loc[i, "timestamp"])
                pre3d = df.loc[i, "timestamp"] + timedelta(days=-3)
                neg_sample_df_pre3d.append(pre3d.strftime("%Y%m%d"))

    neg_sample_df = pd.DataFrame ({
        "channel_id": neg_sample_df_channel,
        "coin": neg_sample_df_symbol,
        "pair": neg_sample_df_pair,
        "timestamp": neg_sample_df_timestamp,
        "pre3d": neg_sample_df_pre3d
    })

    neg_sample_df["timestamp"] = neg_sample_df.timestamp.apply(pd.to_datetime)
    neg_sample_df["timestamp_unix"] = (neg_sample_df["timestamp"].astype(int) / (10 ** 6)).astype(int)

    print("pause")

    for i in range(len(neg_sample_df)):
        try:
            file_name = neg_sample_df.loc[i, "coin"] + neg_sample_df.loc[i, "pair"] + "-1m-" + neg_sample_df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            statistics = pd.read_csv("../../CoinStatistics/data/concat/" + file_name)
        except:
            continue

        statistics["open_scale"] = statistics["open"] * 10 ** 8
        statistics["close_scale"] = statistics["close"] * 10 ** 8
        statistics["high_scale"] = statistics["high"] * 10 ** 8
        statistics["low_scale"] = statistics["low"] * 10 ** 8
        statistics["maker_buy_base_asset_volume"] = statistics["volume"] - statistics["taker_buy_base_asset_volume"]
        statistics["maker_buy_quote_asset_volume"] = statistics["quote_asset_volume"] - statistics[
            "taker_buy_quote_asset_volume"]

        # before pump
        idx = np.max(statistics[statistics.open_time < neg_sample_df.loc[i, "timestamp_unix"]].index)
        idx = idx - 30

        debug_cnt2 = 0
        debug_idx2 = []

        try:
            pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list, pre_volume_tb_q_list = pre_pump_statistics(
                statistics, idx, bucket_num=72, bucket_size_min=60)
            return_rate = []
            W = [1, 3, 6, 12, 24, 36, 48, 60, 72]
            for w in W:

                neg_sample_df.loc[i, "pre_" + str(w) + "h_return"] = pre_price_list[0] / pre_price_list[w] - 1.0
                neg_sample_df.loc[i, "pre_" + str(w) + "h_price"] = pre_price_list[w - 1]
                neg_sample_df.loc[i, "pre_" + str(w) + "h_price_avg"] = np.mean(pre_price_list[:w])
                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume"] = np.sum(pre_volume_list[w - 1])
                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_avg"] = np.mean(pre_volume_list[:w])
                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_sum"] = np.sum(pre_volume_list[:w])

                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_tb"] = pre_volume_tb_list[w - 1]
                if w > 1:
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_tb_avg"] = np.mean(pre_volume_tb_list[:w])
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_tb_sum"] = np.sum(pre_volume_tb_list[:w])

                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote"] = pre_volume_q_list[w - 1]
                if w > 1:
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote_avg"] = np.mean(pre_volume_q_list[:w])
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote_sum"] = np.sum(pre_volume_q_list[:w])

                neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote_tb"] = pre_volume_tb_q_list[w - 1]

                if w > 1:
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_avg"] = np.mean(pre_volume_tb_q_list[:w])
                    neg_sample_df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_sum"] = np.sum(pre_volume_tb_q_list[:w])

            debug_cnt2 += 1

        except:
            debug_idx2.append(i)
            continue

    print("pause")

    neg_coin_gecko_df = pd.read_csv("raw/neg_coin_gecko_statistics.txt",
                                    delimiter="\t",
                                    names=["key", "pre_3d_market_cap_usd", "pre_3d_market_cap_btc", "pre_3d_price_usd", "pre_3d_price_btc", "pre_3d_volume_usd", "pre_3d_volume_btc", 'pre_3d_twitter_index','pre_3d_reddit_index', 'pre_3d_alexa_index'])

    neg_coin_gecko_df[["coin", "date"]] = neg_coin_gecko_df.key.str.split("_", expand=True)

    neg_coin_gecko_df_new = pd.merge(left=neg_sample_df, right=neg_coin_gecko_df, how='left', left_on=["coin", "pre3d"], right_on=["coin", "date"], sort=False)

    print("pause")

# symbol2coinId = dict(
#     {'REN': 'republic-protocol',
#      'SKY': 'skycoin',
#      'MET': 'metronome',
#      'REAP': 'reapchain',
#      'BNT': 'bancor',
#      'SOUL': 'phantasma',
#      'SWP': 'kava-swap',
#      'KNC': 'kyber-network',
#      'DATA': 'streamr',
#      'LOOM': 'loom-network',
#      'MORE': 'legends-room',
#      'CDT': 'blox',
#      'SNX': 'havven',
#      'MANA': 'decentraland',
#      'VIBE': 'vibe',
#      'BAS': 'basis-share',
#      'GAM': 'gamma',
#      'ELF': 'aelf',
#      'GAS': 'gas',
#      'GOD': 'bitcoin-god',
#      'ETHOS': 'ethos',
#      'GRT': 'the-graph',
#      'STORM': 'storm',
#      'AKRO': 'akropolis',
#      'DAI': 'dai',
#      'INK': 'ink',
#      'FTM': 'fantom',
#      'GVT': 'genesis-vision',
#      'ENJ': 'enjincoin',
#      'HC': 'hshare',
#      'SUSD': 'nusd',
#      'BSC': 'bitsonic-token',
#      'BTCST': 'btc-standard-hashrate-token',
#      'CREAM': 'cream-2',
#      'TKN': 'tokencard',
#      'KEEP': 'keep-network',
#      'LRC': 'loopring',
#      'SNT': 'status',
#      'AST': 'airswap',
#      'ADX': 'adex',
#      'FXS': 'frax-share',
#      'LINK': 'chainlink',
#      'ACM': 'ac-milan-fan-token',
#      'SIG': 'signal-token',
#      'ORC': 'orbit-chain',
#      'PHB': 'phoenix-global',
#      'GNT': 'greentrust',
#      'HONOR': 'honor-token',
#      'PAX': 'payperex',
#      'CVC': 'civic',
#      'BOX': 'defibox',
#      'PIE': 'pie-share',
#      'BAT': 'basic-attention-token',
#      'BTS': 'bitshares',
#      'TCT': 'tokenclub'
# })

# cg = CoinGeckoAPI()
# coin_list = cg.get_coins_list()
#
# symbol2id = {}
# for c in coin_list:
#     symbol2id[c["symbol"]] = c["id"]
# for s in symbol2coinId.keys():
#     symbol2id[s.lower()] = symbol2coinId[s]

# 先random产生，然后去读coin_geko数据和binance的数据。


# def pre_pump_statistics(statistics, idx, bucket_num=72, bucket_size_min=60):
#     price_list = []
#     volume_list = []
#     volume_tb_list = []
#     volume_q_list = []
#     volume_tb_q_list = []
#
#     for j in range(bucket_num + 1):
#         C = 0
#         V = 0
#         V_tb = 0
#         V_q = 0
#         V_q_tb = 0
#         prices = []
#
#         for i in range(idx - (j + 1) * bucket_size_min, idx - j * bucket_size_min):
#             p = float(statistics.loc[i, "high"] + statistics.loc[i, "low"]) / 2
#             v = float(statistics.loc[i, "volume"])
#             v_tb = float(statistics.loc[i, "taker_buy_base_asset_volume"])
#             v_q = float(statistics.loc[i, "quote_asset_volume"])
#             v_q_tb = float(statistics.loc[i, "taker_buy_quote_asset_volume"])
#
#             C += p * v
#             V += v
#             V_tb += v_tb
#             V_q += v_q
#             V_q_tb += v_q_tb
#             prices.append(p)
#
#         if V != 0:
#             AggregatedPrice = C / V
#
#         else:
#             AggregatedPrice = np.mean(prices)
#
#         price_list.append(AggregatedPrice)
#         volume_list.append(V)
#         volume_tb_list.append(V_tb)
#         volume_q_list.append(V_q)
#         volume_tb_q_list.append(V_q_tb)
#
#     return price_list, volume_list, volume_tb_list, volume_q_list, volume_tb_q_list


#
#     with open("coin_date_to_statistics_pre3d_dict", "rb") as f:
#         coin_date_to_statistics_pre3d = pkl.load(f)
#
#     debug_cnt3 = 0
#     debug_idx3 = []
#     for i in range(len(df)):
#
#         key = df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")
#         try:
#             market_cap_usd = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["usd"]
#             market_cap_btc = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["btc"]
#             price_usd = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["usd"]
#             price_btc = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["btc"]
#             volume_usd = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["usd"]
#             volume_btc = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["btc"]
#
#             twitter_follower = coin_date_to_statistics_pre3d[key]["community_data"]["twitter_followers"]
#             reddit_subscriber = coin_date_to_statistics_pre3d[key]["community_data"]["reddit_subscribers"]
#             alexa_rank = coin_date_to_statistics_pre3d[key]["public_interest_stats"]["alexa_rank"]
#
#             if market_cap_usd != 0:
#                 df.loc[i, "pre_3d_market_cap_usd"] = market_cap_usd
#
#             if market_cap_btc != 0:
#                 df.loc[i, "pre_3d_market_cap_btc"] = market_cap_btc
#
#             df.loc[i, "pre_3d_price_usd"] = price_usd
#             df.loc[i, "pre_3d_price_btc"] = price_btc
#
#             df.loc[i, "pre_3d_volume_usd"] = volume_usd
#             df.loc[i, "pre_3d_volume_btc"] = volume_btc
#
#             if twitter_follower:
#                 df.loc[i, "pre_3d_twitter_index"] = twitter_follower
#
#             if reddit_subscriber:
#                 df.loc[i, "pre_3d_reddit_index"] = reddit_subscriber
#
#             if alexa_rank:
#                 df.loc[i, "pre_3d_alexa_index"] = alexa_rank
#
#             debug_cnt3 += 1
#         except:
#             debug_idx3.append(key)
#             continue
#
#     print("pause")
#
#     df.to_csv("pump_sample_raw.csv", index = False)
