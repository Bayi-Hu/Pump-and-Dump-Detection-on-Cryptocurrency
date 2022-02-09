import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

import os
import pickle

def load_obj(file):
    with open("./Data/" + file, "rb") as f:
        return pickle.load(f)

def load_session_list(channel_id):
    with open("./Labeled/" + channel_id + "_session.pkl", "rb") as f:
        return pickle.load(f)

class Message(object):

    def __init__(self, m):
        self.message_id = str(m["peer_id"]["channel_id"]) + "-" + str(m["id"])
        self.content = m["message"]
        self.date = m["date"]
        self.pred_pump = False

class Session(object):

    def __init__(self, id, message_list, pred_pump_flag):
        self.id = id
        self.pred_pump_flag = pred_pump_flag
        self.message_list = message_list
        self.target_coin = None
        self.target_exchange = None
        self.target_timestamp = None
        self.session_id = None
        self.first_pred_pump_offset = None
        self.last_pred_pump_offset = None
        self.pred_pump_cnt = 0

    def labeling(self):
        self.target_coin = input("coin")
        self.target_exchange = input("exchange")
        self.target_timestamp = input("timestamp")


def save_session_list(obj, channel_id):
    with open("./Labeled/" + channel_id + "_session.pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


with open("Labeled/pump_attack_browsed.txt", "r") as f:
    browsed_sessions = f.readlines()
    browsed_sessions = list(map(lambda x: "-".join(x[:-1].split("\t")[:2]), browsed_sessions))

def session_list_print(sess_list, channel_id):
    """
    print the session information
    """

    min_datetime = datetime(2019, 1, 1).replace(tzinfo=pytz.timezone('UTC'))

    for sess in sess_list:
        if sess.message_list[0].date <= min_datetime:
            continue
        if str(channel_id) + "-" + str(sess.id) in browsed_sessions:
            continue

        stop_offset = min(sess.last_pred_pump_offset + 6, len(sess.message_list))
        start_offset = max(sess.first_pred_pump_offset - 5, 0)

        delta = 0
        for j in range(start_offset, stop_offset):
            i = j + delta
            message = sess.message_list[i]
            date = message.date
            content = message.content

            print(content)
            print("------------| " + str(message.pred_pump) + " | " + str(i) + "/" + str(stop_offset-1) + " ｜ " + str(date) + " |---------------")

            op = input("")
            if op == "1":
                labeling(str(channel_id), str(sess.id))
                print("------------------------------------")

            elif op == "?":

                print("-------------")
                print("channel:"+ str(channel_id), " | " + "session_index:" + str(sess.id))
                print("-------------")
                delta -= 1

            elif op == "#":
                break
            elif op == "##":
                return

            elif op == "r":
                if j > 0:
                    delta -= 2
                else:
                    print("not illegal")
                    delta -= 1

            else:
                pass

        f = open("Labeled/pump_attack_browsed.txt", "a")
        f.write(str(channel_id) + "-" + str(sess.id) + "\n")
        f.close()


def labeling(channel_id, session_id):

    f = open("Labeled/pump_attack.txt", "a")
    coin = input("Coin:")
    exchange = input("Exchange:")
    pair = input("Pair:")
    timestamp = input("Pump_time:")

    write_list = [channel_id, session_id, coin, exchange, pair, timestamp]
    f.write("\t".join(write_list) + "\n")
    f.close()

if __name__ == '__main__':

    cnt = 0
    for root, dirs, files in os.walk("./Labeled"):

        for file in files:
            if file.endswith(".pkl") == False:
                continue
            channel_id = file.split("_")[0]
            pred_pump_session_list = load_session_list(channel_id)
            session_list_print(pred_pump_session_list, channel_id)

    # print(cnt) 3133

    # file = "1394568162_session.pkl"
    # file = "1095533634_session.pkl"
    #
    # channel_id = file.split("_")[0]
    # pred_pump_session_list = load_session_list(channel_id)
    #
    # print("pause")
    # session_list_print(pred_pump_session_list, channel_id)
