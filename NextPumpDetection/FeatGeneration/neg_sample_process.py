#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
import pickle as pkl
from datetime import *
from CoinStatistics.utility import download_monthly_klines

# markets_coin_list = []
# for i in range(4000):
#     markets = cg.get_coins_markets(vs_currency="usd", order = "market_cap_desc", per_page=100, page=i)
#     markets_coin_list += markets

binance_symbol = "HNT,BOND,WOO,ZEN,GRS,AAVE,LINKDOWN,DCR,UMA,FORTH,RVN,CVX,STEEM,RNDR,KSM,BAR,XTZUP,FTT,ERD,CKB,FXS,MCO,SUB,JUV,BNX,GHST,WING,CHESS,LTC,TRXDOWN,CHAT,XRPBULL,BCHUP,SNM,DGB,STPT,EGLD,TWT,SAND,XMR,BICO,TROY,REN,BAND,ACH,MBOX,GLMR,IOST,DAI,CDT,ATM,SOL,USDC,ARN,BRD,ETHBULL,FILDOWN,RSR,ADA,BURGER,YGG,TKO,SKY,BNBBEAR,OGN,CLOAK,FRONT,PERL,CLV,LOKA,RAY,WAN,EASY,AST,COTI,LINK,PAXG,GVT,KNC,DENT,LUN,NPXS,DOGE,OOKI,HIVE,XLMDOWN,MBL,GXS,FIS,NULS,TLM,ENJ,STRAT,PERP,PUNDIX,C98,FLOW,NAV,XTZDOWN,NAS,RDN,XEC,DYDX,ZRX,VEN,NEO,SRM,ANT,TORN,QNT,ICN,USDP,EOS,UTK,IDEX,GAS,BTT,REEF,ETH,LTO,JST,ALPHA,LRC,EOSUP,RCN,ARPA,ATA,BTTC,HOT,MATIC,FIO,THETA,MDX,DOT,COMP,KEEP,ENG,ASR,UNIDOWN,GALA,ELF,STMX,XRPUP,LINKUP,SNX,SANTOS,DF,TNB,KEY,VIBE,BUSD,DASH,RLC,SXP,NXS,POA,SFP,SPELL,ETHBEAR,BCHDOWN,BAT,BAKE,OM,ROSE,TRXUP,CELO,XVS,KP3R,ETHDOWN,VTHO,MDA,VET,BCHSV,ICP,NU,ACA,AUTO,AION,SNGLS,UST,FARM,BULL,QKC,HARD,AUD,USDSB,BCH,EOSBEAR,WRX,TRX,FIL,NBS,EZ,DODO,OST,REP,AMP,LUNA,LTCDOWN,FOR,CAKE,VITE,MKR,WIN,LOOM,STRAX,SUPER,MASK,LIT,EOSDOWN,MITH,TCT,BEAM,AKRO,ADX,LSK,MANA,GNO,FUN,WAVES,MOVR,SALT,TRB,POLS,UNIUP,ICX,ATOM,GTO,TVK,CHR,VOXEL,WBTC,YOYO,STORJ,EUR,SKL,APPC,AGI,ALPACA,RENBTC,SNT,GLM,MINA,AUDIO,STX,CTK,MTL,GO,NMR,LEND,PYR,BADGER,QUICK,PHX,NKN,GBP,ALPINE,FIRO,BAL,QSP,SUN,VIDT,ERN,FLM,XVG,STORM,DOTUP,AAVEDOWN,ONT,ARDR,RARE,REQ,NEBL,ALGO,BOT,PROM,DEXE,QTUM,HBAR,DOCK,CFX,ALCX,SHIB,ANY,SYS,WTC,CTXC,KLAY,OXT,CELR,BTCUP,DOTDOWN,XEM,SSV,BETA,CVP,1INCH,EOSBULL,FIDA,SUSHI,SXPDOWN,FUEL,ADADOWN,RIF,ENS,WAXP,DEGO,IOTX,BCD,PEOPLE,VGX,ANKR,UNFI,BTC,DNT,SUSHIDOWN,API3,HIGH,1INCHUP,HSR,DIA,OG,1INCHDOWN,AVA,BTCB,INS,AMB,SUSD,FTM,POND,OCEAN,BCPT,SUSHIUP,PNT,ANC,NANO,ONE,POWR,TOMO,BNT,DAR,YFII,IMX,IRIS,BEL,GRT,SC,ZIL,PLA,OAX,YFI,LPT,MLN,EDO,WINGS,AE,XNO,MFT,XLM,PPT,ACM,BTCDOWN,SLP,BTCST,HC,XTZ,AGLD,LTCUP,LAZIO,RAD,BCC,SCRT,PHA,BNB,FLUX,RUNE,EVX,BTG,DREP,XRPBEAR,TRIG,XRPDOWN,FILUP,WPR,PHB,OMG,GTC,DUSK,NEAR,IOTA,RAMP,SXPUP,PIVX,COS,INJ,BKRW,TNT,XRP,AVAX,EPS,MOD,QI,CND,QLC,KMD,TRU,GNT,JOE,RPX,ALICE,COCOS,NCASH,BLZ,USDS,XLMUP,CRV,TRIBE,BNBDOWN,ZEC,BTS,POE,XZC,CTSI,ILV,RGT,AERGO,AUCTION,UNI,LINA,ARK,FET,CHZ,POLY,AAVEUP,AGIX,YFIDOWN,BQX,MC,ETC,MTH,WNXM,BNBUP,BEAR,BCHABC,BNBBULL,ETHUP,TUSD,CVC,ADAUP,BCN,MIR,VIA,CITY,PORTO,KAVA,AXS,DLT,CMT,AR,WABI,PSG,TFUEL,VIB,JASMY,ORN,DGD,PAX,ONG,DATA,YFIUP,MDT,BZRX"
binance_symbol_list = binance_symbol.split(",")

require_binance_file_names = set()
require_coingecko_file_names = set()

if __name__ == '__main__':
    
    # train/validation/test 要根据时间点分，否则会有leakage
    df = pd.read_csv("../../Telegram/Labeled/pump_attack_new.txt", sep="\t")
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)
    df["timestamp_unix"] = (df["timestamp"].astype(int) / (10 ** 6)).astype(int)
    
    try:
        require_binance_file_names = pd.read_csv("neg_binance_required_file_name.txt", names=["file_name"]).file_name.values
    except:
        for i in range(len(df)):
            if df.loc[i, "exchange"] != "binance":
                continue
            for symbol in binance_symbol_list:
                if symbol == df.loc[i, "coin"]:
                    continue
                file_name = symbol + df.loc[i, "pair"] + "-" + df.loc[i, "timestamp"].strftime("%Y%m")
                require_binance_file_names.add(file_name)

        with open("neg_binance_required_file_name.txt", "w") as f:
            for file_name in require_binance_file_names:
                f.write(file_name+"\n")


    last_month_file_names = set()
    for i in range(len(require_binance_file_names)):
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

# for coin in unattack_markets_coin_list:
#     rand_time = random.randint(start, end)
#     random_date = time.strftime("%d-%m-%Y", time.localtime(rand_time))
#
#     try:
#         H = cg.get_coin_history_by_id(coin["id"], date=random_date)
#
#         key = coin["id"] + "_" + time.strftime("%Y%m%d", time.localtime(rand_time))
#
#         unattack_coin_id_date_to_statistics[key] = H
#
#     except:
#         error_key.append(key)
#
#     print(str(i) + ":" + key)
#
#     time.sleep(1.5)
#     i += 1


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
