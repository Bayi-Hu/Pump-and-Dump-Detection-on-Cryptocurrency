# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import re
import os
from urlextract import URLExtract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

punctuations = '''’'!()-[]{};:'"\,<>./?@#$%^&*_~�'''

def remove_punctuation_url(d):
    d=d.lower()
    d = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', d, flags=re.MULTILINE) #This line is for removing url
    review = d.replace('\n', '')
    no_punct = ""
    for char in review:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def remove_stopwords(d):
    text_tokens = word_tokenize(d.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
    ls = ""
    for w in tokens_without_sw:
        ls = ls +" "+w.lower()
    return ls

def load_obj(file):
    with open("./Data/" + file, "rb") as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open("./PreSelection/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

crypto_match = []
exchange_match = []
keyword_match = []
signal_count = 0

def pre_select(message):

    # SYMBOL, EXCHANGE, OR KEYWORDS
    words = re.split('[ ,.?!/\n;#$]', message["message"].lower())
    for w in words:
        if w in ["", "the", "we", "min", "coin", "am", "now", "pump", "buy", "profit", "profits", "for", "a", "in", "sure", "get", "less"]:
            continue
        if w in crypto_symbol_list:
            crypto_match.append(w)
            return True
        if w in exchange_symbol_list:
            exchange_match.append(w)
            return True
        if w in keyword_list:
            keyword_match.append(w)
            return True

    return False

if __name__ == '__main__':

    with open("./Data/crypto_symbol", "r") as f:
        crypto_symbol = f.readlines()[0]
    crypto_symbol_list = crypto_symbol.split(",")

    with open("./Data/exchange", "r") as f:
        exchange_symbol = f.readlines()[0]
    exchange_symbol_list = exchange_symbol.split(",")
    keyword_list = ["target", "buy", "pump"]

    for root, dirs, files in os.walk("./Data"):

        for file in files:
            message_list = []
            if file.endswith(".pkl"):
                all_messgaes = load_obj(file)

                for m in all_messgaes:
                    if m["from_id"] != None or m["_"] != "Message":
                        continue

                    if pre_select(m):
                        message_list.append(m)

            if(len(message_list)>0):
                save_obj(message_list, file[:-4])
            
    print("pause")



