import numpy as np
import json
import datetime
import pandas as pd
import os

if __name__ == '__main__':

    with open("HistoryPumpDump.json", "r") as f:
        history_pump_list = json.load(f)

    key_list = ['currency', 'exchange', 'channelTitle', 'channelLink', 'channelId', 'max', 'priceBeforePump',
                'duration', 'volume', 'signalTime', 'theoreticalBuyPrice', 'theoreticalBuyTime', 'theoreticalProfit']
    history_pump_list_new = []

    for history_pump in history_pump_list:
        event = {}
        for k in key_list:
            if k == "signalTime":
                try:
                    timestamp = datetime.datetime.strptime(history_pump["signalTime"], "%Y-%m-%dT%H:%M:%SZ")

                except:
                    timestamp = datetime.datetime.strptime(history_pump["signalTime"], "%Y-%m-%dT%H:%M:%S.%fZ")

                event["signalTime"] = timestamp

            else:
                event[k] = history_pump[k]

        if event["signalTime"].minute in [0, 30]:
            history_pump_list_new.append(event)


    print("pause")
    print(len(history_pump_list_new))