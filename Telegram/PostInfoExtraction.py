# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import re
import os
from urlextract import URLExtract

def load_obj(file):
    with open("./Data/" + file, "rb") as f:
        return pickle.load(f)


class Message(object):
    def __init__(self, content, timestamp):
        self.content = content
        self.timestamp = timestamp
        self.target_coin = None
        self.target_timestamp = None
        self.urls = []
        # self.get_pump_target(
        self.get_url()

    def get_pump_target(self):
        """
        Analyze content to get target price
        :return:
        """
    def get_url(self):
        """
        :return:
        """
        extractor = URLExtract()
        url = extractor.find_urls(self.content)
        if len(url) > 0:
            self.urls.append(url)

    def get_exchange(self):
        """

        :return:
        """

    def get_coin(self):
        """

        :return:
        """


class Channel(object):
    def __init__(self, message_list):
        self.message_list = message_list

        # sort
        def take_timestamp(m):
            return m.timestamp
        self.message_list.sort(key=take_timestamp)

        # get all urls
        self.urls = []
        for message in self.message_list:
            if(len(message.urls))>0:
                self.urls += message.urls

    def session_split(self):
        threshold = timedelta(hours=12)
        self.session_ids = [0]
        for i in range(len(self.message_list)-1):
            delta = self.message_list[i+1].timestamp - self.message_list[i].timestamp
            if delta >= threshold:
                self.session_ids.append(i+1)

    def session_print(self):
        tmp_sid = 0
        for i in range(len(self.session_ids)-1):
            print("------the "+ str(i) +"th"+ " session-------")
            for id in range(self.session_ids[i], self.session_ids[i+1]):
                print(self.message_list[id].timestamp)
                print(self.message_list[id].content)

        print("------the last session-------")
        for id in range(self.session_ids[-1], len(self.message_list)):
            print(self.message_list[id].timestamp)
            print(self.message_list[id].content)


if __name__ == '__main__':

    all_messgaes = load_obj("1214538537.pkl")
    message_list = []

    for m in all_messgaes:
        try:
            message_list.append(Message(m["message"], m["date"]))
            print(m["date"], m["message"])
        except:
            continue

    channel = Channel(message_list)
    print("pause")
    channel.session_split()
    channel.session_print()
    print(channel.session_ids)





