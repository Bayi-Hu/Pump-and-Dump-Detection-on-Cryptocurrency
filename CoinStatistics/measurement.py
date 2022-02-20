import numpy as np
import pandas as pd

df = pd.read_csv("../Telegram/Labeled/pump_attack_new.txt", sep="\t")
df["timestamp"] = df.timestamp.apply(pd.to_datetime)
df["timestamp_unix"] = df["timestamp"].astype(int) / (10**6)

columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
          "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

for i, row in df.iterrows():
    
    file_name = row.coin + row.pair + "-1m-" + row.timestamp.strftime("%Y-%m") + ".csv"
    statistics = pd.read_csv("./data/unzip/"+file_name, names=columns)

    statistics["open_scale"] = statistics["open"] * 10 ** 8
    statistics["close_scale"] = statistics["close"] * 10 ** 8
    statistics["high_scale"] = statistics["high"] * 10 ** 8
    statistics["low_scale"] = statistics["low"] * 10 ** 8
    statistics["maker_buy_base_asset_volume"] = statistics["volume"] - statistics["taker_buy_base_asset_volume"]
    statistics["maker_buy_quote_asset_volume"] = statistics["quote_asset_volume"] - statistics["taker_buy_quote_asset_volume"]

    idx = np.max(statistics[statistics.open_time < df.loc[i, "timestamp_unix"]].index)

    df_pre_1h = statistics.loc[range(idx - 60, idx), ["open_time", "open_scale", "high_scale", "low_scale", "close_scale", "number_of_trades",
                                          "volume", "taker_buy_base_asset_volume", "maker_buy_base_asset_volume",
                                          "quote_asset_volume", "taker_buy_quote_asset_volume", "maker_buy_quote_asset_volume"]]

    df_1h = statistics.loc[range(idx, idx + 60), ["open_time", "open_scale", "high_scale", "low_scale", "close_scale", "number_of_trades",
                                          "volume", "taker_buy_base_asset_volume", "maker_buy_base_asset_volume",
                                          "quote_asset_volume", "taker_buy_quote_asset_volume", "maker_buy_quote_asset_volume"]]

# ---------------------- #

df = pd.read_csv("../Telegram/Labeled/pump_attack_new.txt", sep="\t")
df["timestamp"] = df.timestamp.apply(pd.to_datetime)
df["timestamp_unix"] = df["timestamp"].astype(int) / (10**6)

coin_date_to_timestamp = {}

for idx, row in df.iterrows():

    attack_date = row.timestamp.strftime("%Y%m%d")
    attack_time = row.timestamp

    try:
        coin_date_to_timestamp[row.coin + "_" + attack_date].append(attack_time)
    except:
        coin_date_to_timestamp[row.coin + "_" + attack_date] = [attack_time]


from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()
# cg.get_coin_history_by_id()

# H = cg.get_coin_history_by_id('appcoins', date='06-06-2019')
# coin_list = cg.get_coins_list()


markets_coin_list = []

for i in range(30):
    markets = cg.get_coins_markets(vs_currency="usd", order = "market_cap_desc", per_page=100, page=i)
    markets_coin_list += markets


for i in range(len(markets_coin_list)):
    print(str(i) + " " + markets_coin_list[i]["symbol"]+ ":", markets_coin_list[i]["market_cap"])


{'id': 'bitcoin',
 'symbol': 'btc',
 'name': 'Bitcoin',
 'localization': {'en': 'Bitcoin', 'de': 'Bitcoin', 'es': 'Bitcoin', 'fr': 'Bitcoin', 'it': 'Bitcoin', 'pl': 'Bitcoin', 'ro': 'Bitcoin', 'hu': 'Bitcoin', 'nl': 'Bitcoin', 'pt': 'Bitcoin', 'sv': 'Bitcoin', 'vi': 'Bitcoin', 'tr': 'Bitcoin', 'ru': 'Биткоин', 'ja': 'ビットコイン', 'zh': '比特币', 'zh-tw': '比特幣', 'ko': '비트코인', 'ar': 'بيتكوين', 'th': 'บิตคอยน์', 'id': 'Bitcoin'},
 'image': {'thumb': 'https://assets.coingecko.com/coins/images/1/thumb/bitcoin.png?1547033579',
           'small': 'https://assets.coingecko.com/coins/images/1/small/bitcoin.png?1547033579'},
 'market_data': {'current_price': {'aed': 50024.57906376443, 'ars': 253468.12429692186, 'aud': 17446.3215245937, 'bch': 5.76928286478153, 'bdt': 1126110.803183989, 'bhd': 5132.860612995706, 'bmd': 13620.3618741461, 'brl': 45117.7211153463, 'btc': 1.0, 'cad': 17128.871750393, 'chf': 13262.4868659029, 'clp': 8362902.190725706, 'cny': 88573.2132675718, 'czk': 289914.5782287119, 'dkk': 84525.1736167662, 'eth': 18.483094024188404, 'eur': 11345.8976447824, 'gbp': 10079.0677868681, 'hkd': 106417.930376984, 'huf': 3526720.3000726495, 'idr': 184652192.175199, 'ils': 47387.96303252911, 'inr': 869671.001953725, 'jpy': 1535062.45448282, 'krw': 14537693.2463698, 'kwd': 4104.645874754543, 'lkr': 2087919.548829924, 'ltc': 60.96840666846534, 'mmk': 18414729.253845528, 'mxn': 267888.750532982, 'myr': 55317.8739192755, 'ngn': 4884546.501733771, 'nok': 111755.75019546246, 'nzd': 19178.1505368914, 'php': 680527.760679833, 'pkr': 1505414.7676248574, 'pln': 47450.61669715, 'rub': 785377.30638701, 'sar': 51079.0811004227, 'sek': 111446.704184538, 'sgd': 18213.1478981081, 'thb': 442954.59869004245, 'try': 51700.07425935065, 'twd': 404053.46952093, 'uah': 382908.08925747185, 'usd': 13620.3618741461, 'vef': 140859.73944813784, 'vnd': 309201434.91677517, 'xag': 804.154745877564, 'xau': 10.4549897745945, 'xdr': 9563.95932114975, 'zar': 168771.061713303, 'bits': 1000000.0, 'link': 22041.447552365687, 'sats': 100000000.0},
                 'market_cap': {'aed': 839030999274.6053, 'ars': 4251262431254.5815, 'aud': 292616246981.057, 'bch': 96764575.68919012, 'bdt': 18887552682553.043, 'bhd': 86090263023.8938, 'bmd': 228445816988.881, 'brl': 756731337692.006, 'btc': 16772375.0, 'cad': 287291860324.498, 'chf': 222443403147.498, 'clp': 140265731631172.94, 'cny': 1485583147878.69, 'czk': 4862556024018.788, 'dkk': 1417687908840.51, 'eth': 310005384.13394696, 'eur': 190297650009.907, 'gbp': 169049904571.772, 'hkd': 1784881435006.67, 'huf': 59151475392930.96, 'idr': 3097055811734500, 'ils': 794808686467.7148, 'inr': 14586448171393.6, 'jpy': 25746643135006.3, 'krw': 243831642763082.0, 'kwd': 68844659853.58617, 'lkr': 35019369642806.27, 'ltc': 1022584979.7960014, 'mmk': 308858744568967.1, 'mxn': 4493130582220.62, 'myr': 927812125576.808, 'ngn': 81925445632016.88, 'nok': 1874409350684.6182, 'nzd': 321663132611.194, 'php': 11414066800032.4, 'pkr': 25249381013141.95, 'pln': 795859537225.861, 'rub': 13172642699212.8, 'sar': 856717502871.7015, 'sek': 1869225915097.14, 'sgd': 305477746477.531, 'thb': 7429400637203.895, 'try': 867133033005.6757, 'twd': 6776936310856.11, 'uah': 6422278063559.784, 'usd': 228445816988.881, 'vef': 2362552372426.4595, 'vnd': 5186042416962243, 'xag': 13487584955.8882, 'xau': 175355009.120664, 'xdr': 160410312219.069, 'zar': 2830691536203.66, 'bits': 16772375000000.0, 'link': 369687423891.10944, 'sats': 1677237500000000},
                 'total_volume': {'aed': 13223772038.888288, 'ars': 67003156399.47071, 'aud': 4611856472.88116, 'bch': 1525083.9259334763, 'bdt': 297682315984.16693, 'bhd': 1356848571.721612, 'bmd': 3600481281.03768, 'brl': 11926666253.0629, 'btc': 264345.493482963, 'cad': 4527940055.66402, 'chf': 3505878635.37842, 'clp': 2210695506557.1357, 'cny': 23413929770.588, 'czk': 76637612249.77382, 'dkk': 22343848731.4572, 'eth': 4885922.610916088, 'eur': 2999236911.91719, 'gbp': 2664356147.96788, 'hkd': 28131100320.9394, 'huf': 932272618099.0865, 'idr': 48811974863263.9, 'ils': 12526794472.986298, 'inr': 229893610179.28, 'jpy': 405786842057.429, 'krw': 3842973695315.56, 'kwd': 1085044639.3347962, 'lkr': 551932123488.1709, 'ltc': 16116723.547645444, 'mmk': 4867850691962.943, 'mxn': 70815183958.1755, 'myr': 14623030679.6192, 'ngn': 1291207855441.2922, 'nok': 29542128934.978218, 'nzd': 5069657667.76511, 'php': 179894446725.766, 'pkr': 397949609644.3324, 'pln': 12543356686.879, 'rub': 207610951627.194, 'sar': 13502524900.147509, 'sek': 29460434014.7115, 'sgd': 4814563569.00357, 'thb': 117093051981.26692, 'try': 13666681643.19386, 'twd': 106809713794.014, 'uah': 101220027813.38469, 'usd': 3600481281.03768, 'vef': 37235637336.29954, 'vnd': 81736005898715.08, 'xag': 212574683.135671, 'xau': 2763729.43132451, 'xdr': 2528189546.40031, 'zar': 44613869594.2467, 'bits': 264345493482.963, 'link': 5826557330.308955, 'sats': 26434549348296.3}},
 'community_data': {'facebook_likes': None, 'twitter_followers': 603664, 'reddit_average_posts_48h': 2.042, 'reddit_average_comments_48h': 445.896, 'reddit_subscribers': 612412, 'reddit_accounts_active_48h': '14074.0'},
 'developer_data': {'forks': 13660, 'stars': 23665, 'subscribers': 2513, 'total_issues': 3591, 'closed_issues': 3022, 'pull_requests_merged': 5038, 'pull_request_contributors': 450, 'code_additions_deletions_4_weeks': {'additions': None, 'deletions': None}, 'commit_count_4_weeks': 147},
 'public_interest_stats': {'alexa_rank': 2912, 'bing_matches': None}
 }



H["market_data"]["current_price"]["btc"]
H["market_data"]["current_price"]["usd"]


H["market_data"]["market_cap"]["btc"]
H["market_data"]["market_cap"]["usd"]


H["market_data"]["total_volume"]["btc"]
H["market_data"]["total_volume"]["usd"]


H["community_data"]["twitter_followers"]
H["community_data"]["reddit_subscribers"]

H["public_interest_stats"]["alexa_rank"]


{'id': 'appcoins',
 'symbol': 'appc',
 'name': 'AppCoins',
 'localization': {'en': 'AppCoins', 'de': 'AppCoins', 'es': 'AppCoins', 'fr': 'AppCoins', 'it': 'AppCoins', 'pl': 'AppCoins', 'ro': 'AppCoins', 'hu': 'AppCoins', 'nl': 'AppCoins', 'pt': 'AppCoins', 'sv': 'AppCoins', 'vi': 'AppCoins', 'tr': 'AppCoins', 'ru': 'AppCoins', 'ja': 'AppCoins', 'zh': 'AppCoins', 'zh-tw': 'AppCoins', 'ko': '앱코인', 'ar': 'AppCoins', 'th': 'AppCoins', 'id': 'AppCoins'},
 'image': {'thumb': 'https://assets.coingecko.com/coins/images/1876/thumb/appcoins.png?1547036186',
           'small': 'https://assets.coingecko.com/coins/images/1876/small/appcoins.png?1547036186'},
 'market_data': {'current_price': {'aed': 0.3345587663817587, 'ars': 4.085673882979019, 'aud': 0.13067232411328167, 'bch': 0.00022472662549307605, 'bdt': 7.659679328736249, 'bhd': 0.03434145277222783, 'bmd': 0.09108148125065395, 'bnb': 0.002944922033240937, 'brl': 0.35388798725129106, 'btc': 1.1148046577164987e-05, 'cad': 0.12244903257856625, 'chf': 0.09040793369680533, 'clp': 64.12101004188355, 'cny': 0.6288083302582644, 'czk': 2.089954302555289, 'dkk': 0.6048428798301118, 'eos': 0.013551787864157672, 'eth': 0.00036234146552259694, 'eur': 0.08099375179473778, 'gbp': 0.07193296603992276, 'hkd': 0.7137372574504369, 'huf': 26.16588793368783, 'idr': 1299.836793940497, 'ils': 0.3301066124967454, 'inr': 6.3002608966258595, 'jpy': 9.840078908395657, 'krw': 107.33588239464572, 'kwd': 0.027716550151980206, 'lkr': 16.08157722117046, 'ltc': 0.0008529184295296031, 'mmk': 139.12696261037357, 'mxn': 1.8033225205261374, 'myr': 0.3808551189755413, 'ngn': 27.933779484763058, 'nok': 0.7924316572510018, 'nzd': 0.1382116848052863, 'php': 4.707904946231732, 'pkr': 13.375315521658543, 'pln': 0.3466761555658639, 'rub': 5.9511729034364835, 'sar': 0.3415692169121399, 'sek': 0.8605907531967857, 'sgd': 0.12447577769935642, 'thb': 2.853127400176736, 'try': 0.5315272969048036, 'twd': 2.856778674597117, 'uah': 2.4400728827050226, 'usd': 0.09108148125065395, 'vef': 22632.622527792864, 'vnd': 2118.81647913062, 'xag': 0.006162910022899876, 'xau': 6.874556960355606e-05, 'xdr': 0.06557802893010216, 'xlm': 0.7058829198031644, 'xrp': 0.21447027724745815, 'zar': 1.3173088219652571, 'bits': 11.148046577164987, 'link': 0.09974570030311003, 'sats': 1114.8046577164987},
                 'market_cap': {'aed': 35641520.368057474, 'ars': 435244675.79396474, 'aud': 13918138.09358651, 'bch': 23932.743164304688, 'bdt': 816007961.054065, 'bhd': 3658495.0431500045, 'bmd': 9703175.631164787, 'bnb': 314099.9367086759, 'brl': 37702698.04515637, 'btc': 1187.1693926529638, 'cad': 13045182.194753133, 'chf': 9630945.191766389, 'clp': 6831035644.340013, 'cny': 66988783.92243549, 'czk': 222654055.4649813, 'dkk': 64435140.97252807, 'eos': 1442454.2646073315, 'eth': 38552.83864491843, 'eur': 8628781.806228427, 'gbp': 7662374.622891316, 'hkd': 76036306.27302685, 'huf': 2788023157.278208, 'idr': 138487188446.47534, 'ils': 35167219.44003053, 'inr': 671184406.661718, 'jpy': 1048382995.5368588, 'krw': 11434804354.302448, 'kwd': 2952724.860441603, 'lkr': 1713217286.9886818, 'ltc': 90873.13999988907, 'mmk': 14821600776.604244, 'mxn': 192140896.29420927, 'myr': 40573605.72867599, 'ngn': 2975866934.3219285, 'nok': 84423352.864756, 'nzd': 14716719.15120697, 'php': 501650541.44035786, 'pkr': 1424911341.436543, 'pln': 36934167.72246562, 'rub': 633986089.3890444, 'sar': 36388364.09321262, 'sek': 91647027.51101106, 'sgd': 13261097.258897718, 'thb': 303903460.76808167, 'try': 56621076.60499476, 'twd': 304369834.20161694, 'uah': 259948075.15890408, 'usd': 9703175.631164787, 'vef': 2411119234838.465, 'vnd': 225776228560.11948, 'xag': 656596.0411177593, 'xau': 7322.695553571133, 'xdr': 6986218.532209228, 'xlm': 75115446.12776959, 'xrp': 22809693.848050445, 'zar': 140327413.3064861, 'bits': 1187169392.6529639, 'link': 10622044.106914856, 'sats': 118716939265.29637},
                 'total_volume': {'aed': 4616653.785545996, 'ars': 56379158.74198993, 'aud': 1803177.6190114727, 'bch': 3101.0546742384035, 'bdt': 105697686.39309126, 'bhd': 473885.64842191193, 'bmd': 1256854.4227866775, 'bnb': 40637.660163374116, 'brl': 4883382.17429536, 'btc': 153.83447275503215, 'cad': 1689702.5174501757, 'chf': 1247559.9843301696, 'clp': 884820645.8446422, 'cny': 8677071.564034661, 'czk': 28839762.732446793, 'dkk': 8346366.771456615, 'eos': 187004.25465040514, 'eth': 5000.033675866604, 'eur': 1117651.5111909392, 'gbp': 992619.633212121, 'hkd': 9849025.4705621, 'huf': 361069138.57815623, 'idr': 17936748512.785515, 'ils': 4555217.48450576, 'inr': 86938757.07667321, 'jpy': 135785524.4201816, 'krw': 1481152663.077189, 'kwd': 382467.0851260993, 'lkr': 221913403.0131779, 'ltc': 11769.618650364802, 'mmk': 1919845130.8066454, 'mxn': 24884464.487316642, 'myr': 5255507.861230776, 'ngn': 385464682.9244461, 'nok': 10934947.691849789, 'nzd': 1907215.0007120697, 'php': 64965469.07759192, 'pkr': 184569071.98622373, 'pln': 4783864.441099105, 'rub': 82121611.13045879, 'sar': 4713392.613613458, 'sek': 11875490.818908175, 'sgd': 1717670.0420660332, 'thb': 39370964.79379269, 'try': 7334668.088106587, 'twd': 39421349.57389343, 'uah': 33671129.986455135, 'usd': 1256854.4227866775, 'vef': 312312792158.4347, 'vnd': 29238038581.520054, 'xag': 85043.42060711095, 'xau': 948.6360126867004, 'xdr': 904926.3864254493, 'xlm': 9740641.649015902, 'xrp': 2959524.952969705, 'zar': 18177849.06798553, 'bits': 153834472.75503215, 'link': 1376413.985131782, 'sats': 15383447275.503216}},
 'community_data': {'facebook_likes': None, 'twitter_followers': 25574, 'reddit_average_posts_48h': 0.0, 'reddit_average_comments_48h': 0.042, 'reddit_subscribers': 3376, 'reddit_accounts_active_48h': '132.32'},
 'developer_data': {'forks': 2, 'stars': 6, 'subscribers': 13, 'total_issues': 0, 'closed_issues': 0, 'pull_requests_merged': 29, 'pull_request_contributors': 4, 'code_additions_deletions_4_weeks': {'additions': 0, 'deletions': 0}, 'commit_count_4_weeks': 0},
 'public_interest_stats': {'alexa_rank': 1192397, 'bing_matches': None}
 }


{'id': 'appcoins',
 'symbol': 'appc',
 'name': 'AppCoins',
 'localization': {'en': 'AppCoins', 'de': 'AppCoins', 'es': 'AppCoins', 'fr': 'AppCoins', 'it': 'AppCoins', 'pl': 'AppCoins', 'ro': 'AppCoins', 'hu': 'AppCoins', 'nl': 'AppCoins', 'pt': 'AppCoins', 'sv': 'AppCoins', 'vi': 'AppCoins', 'tr': 'AppCoins', 'ru': 'AppCoins', 'ja': 'AppCoins', 'zh': 'AppCoins', 'zh-tw': 'AppCoins', 'ko': '앱코인', 'ar': 'AppCoins', 'th': 'AppCoins', 'id': 'AppCoins'},
 'image': {'thumb': 'https://assets.coingecko.com/coins/images/1876/thumb/appcoins.png?1547036186',
           'small': 'https://assets.coingecko.com/coins/images/1876/small/appcoins.png?1547036186'},
 'market_data': {'current_price': {'aed': 0.3262409150428635, 'ars': 3.9890378985735, 'aud': 0.12740452192072815, 'bch': 0.00022275666519567885, 'bdt': 7.526357333804588, 'bhd': 0.03348329827575585, 'bmd': 0.08881699950066808, 'bnb': 0.002867832932258532, 'brl': 0.3446898933621424, 'btc': 1.1367634319623187e-05, 'cad': 0.11921461757977175, 'chf': 0.08830043983157218, 'clp': 61.54892309406706, 'cny': 0.6136455312500658, 'czk': 2.0297348895887644, 'dkk': 0.5906955738470908, 'eos': 0.013791426236188009, 'eth': 0.00035980216346150545, 'eur': 0.0790985545983054, 'gbp': 0.06999827601246754, 'hkd': 0.6964096522347631, 'huf': 25.414096237121186, 'idr': 1268.8840633662978, 'ils': 0.3200875845004581, 'inr': 6.163947282441095, 'jpy': 9.619094206721151, 'krw': 104.78158604540467, 'kwd': 0.02702994390903683, 'lkr': 15.692975012112651, 'ltc': 0.0008529948334834816, 'mmk': 135.87627235479454, 'mxn': 1.7562319989604134, 'myr': 0.37081097291528925, 'ngn': 27.2401737468549, 'nok': 0.7758116057033633, 'nzd': 0.13401481592556452, 'php': 4.59536926109769, 'pkr': 13.081803987325667, 'pln': 0.3382817468481698, 'rub': 5.8007892262877805, 'sar': 0.33306818897748014, 'sek': 0.8384503275032056, 'sgd': 0.12135661715672967, 'thb': 2.78530110434095, 'try': 0.5082535033025831, 'twd': 2.7820118555814464, 'uah': 2.3932564737210344, 'usd': 0.08881699950066808, 'vef': 22069.926796841104, 'vnd': 2085.361895187895, 'xag': 0.005997510886651809, 'xau': 6.670423113498675e-05, 'xdr': 0.06390347587273262, 'xlm': 0.7279246497471695, 'xrp': 0.22032404239333767, 'zar': 1.318956867259779, 'bits': 11.367634319623187, 'link': 0.08218878517475582, 'sats': 1136.7634319623187},
                 'market_cap': {'aed': 34760078.38939544, 'ars': 424985132.0708222, 'aud': 13571944.650836151, 'bch': 23726.471577765584, 'bdt': 801912816.1017925, 'bhd': 3567553.973565405, 'bmd': 9463208.698236043, 'bnb': 305409.3220067131, 'brl': 36725766.636984274, 'btc': 1210.8638120351497, 'cad': 12700383.129728673, 'chf': 9407631.27355131, 'clp': 6557956311.834083, 'cny': 65382255.21698268, 'czk': 216245381.24566156, 'dkk': 62931388.25943522, 'eos': 1470809.6066818915, 'eth': 38341.677807532775, 'eur': 8426987.345779194, 'gbp': 7457831.753366744, 'hkd': 74200782.82265136, 'huf': 2708417645.4786444, 'idr': 135196131067.3487, 'ils': 34104457.827572905, 'inr': 656751746.4742354, 'jpy': 1024909159.207444, 'krw': 11164191788.221622, 'kwd': 2879966.692760268, 'lkr': 1672043623.0758758, 'ltc': 90910.45371450557, 'mmk': 14477245681.127836, 'mxn': 186918765.0049065, 'myr': 39508896.31513547, 'ngn': 2902366107.7489944, 'nok': 82653368.14795928, 'nzd': 14278969.362307461, 'php': 489624042.7813503, 'pkr': 1393830482.649306, 'pln': 36039684.006362215, 'rub': 618060140.1783226, 'sar': 35487505.778820045, 'sek': 89324409.80373737, 'sgd': 12929714.529321665, 'thb': 296993341.78544056, 'try': 54084480.49088044, 'twd': 296415764.30575144, 'uah': 254994940.23164156, 'usd': 9463208.698236043, 'vef': 2351490417459.1953, 'vnd': 222176490373.80722, 'xag': 638780.2141514582, 'xau': 7107.248260723202, 'xdr': 6808740.8055460425, 'xlm': 77538055.19522856, 'xrp': 23459597.021671206, 'zar': 140494430.20615304, 'bits': 1210863812.0351498, 'link': 8754629.408816319, 'sats': 121086381203.51497},
                 'total_volume': {'aed': 1095488.8957094634, 'ars': 13394845.71356518, 'aud': 427813.4120884041, 'bch': 747.9976971468312, 'bdt': 25272859.73079473, 'bhd': 112434.03191165965, 'bmd': 298239.8350937412, 'bnb': 9629.935998758338, 'brl': 1157438.9760152989, 'btc': 38.17153702501367, 'cad': 400312.4186545742, 'chf': 296505.27221283596, 'clp': 206675982.9421376, 'cny': 2060568.8446461672, 'czk': 6815674.951397257, 'dkk': 1983504.864217284, 'eos': 46310.42153543217, 'eth': 1208.1846774875967, 'eur': 265606.1340979487, 'gbp': 235048.18235440916, 'hkd': 2338483.634978269, 'huf': 85338346.41372319, 'idr': 4260803404.0667443, 'ils': 1074826.5416943352, 'inr': 20698004.113817405, 'jpy': 32300089.916256383, 'krw': 351847541.78514075, 'kwd': 90764.22373358358, 'lkr': 52695658.55698011, 'ltc': 2864.2831879499963, 'mmk': 456260820.4293298, 'mxn': 5897275.7996892845, 'myr': 1245151.3115163695, 'ngn': 91470157.42325042, 'nok': 2605108.5563529, 'nzd': 450010.2100550898, 'php': 15430854.209551556, 'pkr': 43927571.13889336, 'pln': 1135920.9719132876, 'rub': 19478550.63769189, 'sar': 1118414.2935932835, 'sek': 2815443.989491768, 'sgd': 407505.06875753094, 'thb': 9352801.228539722, 'try': 1706671.4915272323, 'twd': 9341756.214486873, 'uah': 8036349.123167812, 'usd': 298239.8350937412, 'vef': 74108913444.78845, 'vnd': 7002465645.407567, 'xag': 20139.12503083938, 'xau': 223.98706335045244, 'xdr': 214582.36839060622, 'xlm': 2444308.282443522, 'xrp': 739829.1592825224, 'zar': 4428943.567096692, 'bits': 38171537.02501367, 'link': 275982.86223224323, 'sats': 3817153702.501367}},
 'community_data': {'facebook_likes': None, 'twitter_followers': 25563, 'reddit_average_posts_48h': 0.043, 'reddit_average_comments_48h': 0.0, 'reddit_subscribers': 3374, 'reddit_accounts_active_48h': '150.0'},
 'developer_data': {'forks': 2, 'stars': 6, 'subscribers': 13, 'total_issues': 0, 'closed_issues': 0, 'pull_requests_merged': 29, 'pull_request_contributors': 4, 'code_additions_deletions_4_weeks': {'additions': 0, 'deletions': 0}, 'commit_count_4_weeks': 0},
 'public_interest_stats': {'alexa_rank': 1192397, 'bing_matches': None}
 }




