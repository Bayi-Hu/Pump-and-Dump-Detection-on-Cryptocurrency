import numpy as np
import pandas as pd
import pickle
import os
import re
import emoji
from urlextract import URLExtract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

punctuations = '''’'!()-[]{};:'"\,<>./?@#$%^&*_~�=‘'''

def remove_punctuation_url_emoji(d):
    d=d.lower()

    def filter_emoji(desstr, restr=''):
        try:
            co = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return co.sub(restr, desstr)

    d = filter_emoji(d)
    d = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', d, flags=re.MULTILINE) #This line is for removing url
    review = d.replace('\n', ' ')
    no_punct = ""
    for char in review:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def remove_stopwords(d):
    text_tokens = word_tokenize(d.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
    ls = " ".join(tokens_without_sw)
    # for w in tokens_without_sw:
    #     ls = ls +" "+w.lower()
    return ls

def load_obj(file):
    with open(file, "rb") as f:
        return pickle.load(f)

preselect_sample = "./preselect_sample"
f_write = open(preselect_sample, "w")

with open("raw/crypto_symbol", "r") as f:
    crypto_symbol = f.readlines()[0]
crypto_symbol_list = crypto_symbol.split(",")

with open("raw/exchange", "r") as f:
    exchange_symbol = f.readlines()[0]
exchange_symbol_list = exchange_symbol.split(",")
keyword_list = ["target", "buy", "pump", "left", "announcement", "hold"]

for root, dirs, files in os.walk("PreSelection"):

    cnt = 0
    for file in files:
        cnt += 1
        if file.endswith(".pkl")==False:
            continue

        all_messages = load_obj(file)

        for i in range(len(all_messages)):
            m = all_messages[i]
            sample_id = m["id"]
            channel_id = m["peer_id"]["channel_id"]
            date = m["date"]

            # weekday ?
            weekday = date.weekday()

            # near 整点
            on_time = 0
            if date.minute in [58,59,0,1,2,28,29,30,31,32]:
                on_time = 1

            message = remove_punctuation_url_emoji(m["message"])
            message_wo_stop = remove_stopwords(message)

            # print(m["message"])
            # print(message)
            # print(message_wo_stop)

            word_list = message_wo_stop.split(" ")
            length = len(word_list)
            crypto_signal = 0
            exchange_signal = 0
            keyword_signal = 0
            for w in word_list:
                if w in ["", "the", "we", "min", "coin", "am", "now", "pump", "buy", "profit", "profits", "for", "a",
                         "in", "sure", "get", "less"]:
                    continue
                if w in crypto_symbol_list:
                    crypto_signal += 1
                if w in exchange_symbol_list:
                    exchange_signal += 1
                if w in keyword_list:
                    keyword_signal += 1

            arr = "\t".join([str(channel_id), str(sample_id), message, message_wo_stop, str(date), str(weekday), str(on_time), str(crypto_signal), str(exchange_signal), str(keyword_signal), str(length)]) + "\n"
            f_write.writelines(arr)

            if i % 100 == 0:
                print(str(cnt)+"/"+str(len(files))+" : "+str(i+1)+"/"+str(len(all_messages)))

