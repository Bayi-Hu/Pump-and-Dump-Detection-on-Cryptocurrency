import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle

def load_obj(file):
    with open("./Data/" + file, "rb") as f:
        return pickle.load(f)

def load_session_list(channel_id):
    with open("./Labeled/" + channel_id + "_session.pkl", "rb") as f:
        return pickle.load(f)


def save_session_list(obj, channel_id):
    with open("./Labeled/" + channel_id + "_session.pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


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


def session_split(message_list, threshold=12):

    session_list = []
    threshold = timedelta(hours=threshold)
    session_message_list = []
    session_id = 0
    session_pred_pump = False

    for i in range(len(message_list)):

        session_message_list.append(message_list[i])
        if message_list[i].pred_pump:
            session_pred_pump = True

        if i == len(message_list) - 1:
            session_list.append(Session(session_id, session_message_list, session_pred_pump))

        else:
            delta = message_list[i + 1].date - message_list[i].date
            if delta >= threshold:
                session_list.append(Session(session_id, session_message_list, session_pred_pump))
                session_message_list = []
                session_pred_pump = False
                session_id += 1

    return session_list

if __name__ == '__main__':

    pred_pump_sample = pd.read_csv("./Labeled/pred_pump_annouce.csv", sep="\t")
    channels = set(pred_pump_sample.channel_id.values)
    pred_pump_sample["id"] = pred_pump_sample.channel_id.astype(str) + "-" + pred_pump_sample.sample_id.astype(str)

    pred_pump_sample_ids = set(pred_pump_sample.id.values)

    # all_messages = load_obj("1214538537.pkl")
    # all_messages.reverse()

    all_messages = load_obj("1095533634.pkl")
    all_messages.reverse()

    message_list = []
    for m in all_messages:
        if m["_"] != "Message" or m["from_id"] != None:
            continue
        m_obj = Message(m)
        if m_obj.message_id in pred_pump_sample_ids:
            m_obj.pred_pump = True

        message_list.append(m_obj)


    pred_pump_session_list = load_session_list("1095533634")
    print("pause")

    session_list = session_split(message_list)

    print("pause")

    # pred_pump_session_list = []
    #
    # for session in session_list:
    #     if session.pred_pump_flag:
    #         for i in range(len(session.message_list)):
    #             m = session.message_list[i]
    #             if m.pred_pump:
    #                 if session.pred_pump_cnt == 0:
    #                     session.first_pred_pump_offset = i
    #                 session.pred_pump_cnt += 1
    #                 session.last_pred_pump_offset = i
    #
    #         if session.pred_pump_cnt > 1:
    #             pred_pump_session_list.append(session)


    # if len(pred_pump_session_list)>0:
    #     save_session_list(pred_pump_session_list, str(c))
