import pdb
import time
import json
import random
import pytz
import datetime
from datetime import timezone
import numpy as np

import argparse
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

if __name__ == "__main__":
    # init_top()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--start_time',
                        type=int,
                        help='start_time UTC timestamp')
    parser.add_argument('--end_time', type=int, help='end_time UTC timestamp')
    parser.add_argument('--interval',
                        type=int,
                        help='accept any interval larger than 60 seconds')
    parser.add_argument('--symbol', type=str, help='symbol')
    parser.add_argument('--output_filename', type=str, help='output_filename')
    args = parser.parse_args()

    # start_time = 1630000560
    # end_time = 1630001160
    # interval = 60

    start_time = args.start_time
    end_time = args.end_time
    interval = args.interval
    symbol = args.symbol
    output_filename = args.output_filename

    # connect influxdb
    bucket_trades = "Klines"
    client = InfluxDBClient(
        url="http://localhost:8086",
        token=
        "-vKh8eaHviakWO6CO7CP4AsJAcK7WF845TcCb4mHz22SsYdW0JQ6HtQJ1kwun8eoFC4ariy_SHVquwsMy3nw5Q==",
        org="NUS")

    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    output = open(output_filename, 'w')

    start_timestamp = start_time
    end_timestamp = end_time
    starts = []
    prices = []
    volumes = []

    while start_timestamp < end_timestamp:

        try:

            start = start_timestamp - 1
            stop = start_timestamp + interval - 1

            q_str = '''from (bucket: "{}") |> range(start: {}, stop: {})'''.format(
                bucket_trades, start, stop)
            q_str += '''|> filter (fn: (r) => r["_measurement"] == "{}")'''.format(
                symbol)
            q_str += '''|> filter(fn: (r) => r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "volume")'''
            q_str += '''|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''

            tables = query_api.query(q_str)
            P = 0
            Q = 0
            for table in tables:
                for row in table.records:
                    p = float(row.values["high"] + row.values["low"]) / 2
                    q = float(row.values["volume"])
                    P += p * q
                    Q += q

            if Q != 0:
                AggregatedPrice = P / Q

                starts.append(start_timestamp)
                prices.append(AggregatedPrice)
                volumes.append(Q)

        except Exception as e:
            print(e)

        start_timestamp += interval

    # output the result
    if len(prices) < 1:
        output.write("no data")
    else:
        data = []
        for i in range(len(starts)):
            item = {}
            item["timestamp"] = starts[i]
            item["price"] = prices[i]
            item["volume"] = volumes[i]
            data.append(item)

        result = {}
        result["data"] = data

        json_str = json.dumps(result)

        output.write(json_str)