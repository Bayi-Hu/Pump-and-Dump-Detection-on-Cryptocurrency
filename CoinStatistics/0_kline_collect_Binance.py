"""
  script to download klines.
  set the absoluate path destination folder for STORE_DIRECTORY, and run

  e.g. STORE_DIRECTORY=/data/ ./download-kline.py

"""
import sys
from datetime import *
import pandas as pd
from enums import *
from utility import download_file, get_all_symbols, get_parser, get_start_end_date_objects, convert_to_date_object, \
  get_path

def download_monthly_klines(trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder, checksum):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date
    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)
    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    print("Found {} symbols".format(num_symbols))
    for symbol in symbols:
        print("[{}/{}] - start download monthly {} klines ".format(current+1, num_symbols, symbol))

        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object('{}-{}-01'.format(year, month))

                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
                        download_file(path, file_name, date_range, folder)

                    if checksum == 1:
                        checksum_path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, year, '{:02d}'.format(month))
                        download_file(checksum_path, checksum_file_name, date_range, folder)

        current += 1

def download_daily_klines(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder, checksum):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date
    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)
    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    #Get valid intervals for daily
    intervals = list(set(intervals) & set(DAILY_INTERVALS))
    print("Found {} symbols".format(num_symbols))

    for symbol in symbols:
        print("[{}/{}] - start download daily {} klines ".format(current+1, num_symbols, symbol))
        
        for interval in intervals:
            for date in dates:
                current_date = convert_to_date_object(date)
                
                if current_date >= start_date and current_date <= end_date:
                    path = get_path(trading_type, "klines", "daily", symbol, interval)
                    file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date)
                    download_file(path, file_name, date_range, folder)
            
                if checksum == 1:
                    checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
                    checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, date)
                    download_file(checksum_path, checksum_file_name, date_range, folder)

        current += 1

if __name__ == "__main__":

    df = pd.read_csv("../Telegram/Labeled/pump_attack_new.txt", sep="\t")

    df["timestamp"] = df.timestamp.apply(pd.to_datetime)

    for i, row in df.iterrows():

        print(row)
        # row.timestamp.strftime("%Y%m%d")

        try:
            # download_daily_klines(trading_type="spot",
            #                       symbols=[row.coin+row.pair],
            #                       num_symbols = 1,
            #                       intervals=["1m"],
            #                       dates=[row.timestamp.strftime("%Y-%m-%d")],
            #                       start_date="",
            #                       end_date="",
            #                       folder="",
            #                       checksum=0)

            # download_monthly_klines(trading_type="spot",
            #                         symbols=[row.coin + row.pair],
            #                         num_symbols=1,
            #                         intervals=["1m"],
            #                         years=[row.timestamp.year],
            #                         months=[row.timestamp.month],
            #                         start_date=None,
            #                         end_date=None,
            #                         folder="",
            #                         checksum=0
            #                         )

            if row.timestamp.day > 3:
                last_month_timestamp = row.timestamp - timedelta(days=15)

            else:
                last_month_timestamp = row.timestamp - timedelta(days=31)
            # print(last_month_timestamp.year, last_month_timestamp.month)

            download_monthly_klines(trading_type="spot",
                                    symbols=[row.coin + row.pair],
                                    num_symbols=1,
                                    intervals=["1m"],
                                    years=[last_month_timestamp.year],
                                    months=[last_month_timestamp.month],
                                    start_date=None,
                                    end_date=None,
                                    folder="",
                                    checksum=0
                                    )

        except:
            continue
