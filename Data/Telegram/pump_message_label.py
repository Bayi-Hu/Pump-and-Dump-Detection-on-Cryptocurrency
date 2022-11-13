# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import pytz
import re
import os
from urlextract import URLExtract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_obj(file):
    with open("./PreSelection/" + file, "rb") as f:
        return pickle.load(f)

with open("Labeled/label.txt", "r") as f:
    labeled_samples = f.readlines()

labeled_samples = list(map(lambda x: "-".join(x[:-1].split(" ")[1:]), labeled_samples))

f = open("Labeled/label.txt", "a")

for root, dirs, files in os.walk("PreSelection"):

    for file in files:

        min_datetime = datetime(2021, 1, 1).replace(tzinfo=pytz.timezone('UTC'))
        if file.endswith(".pkl"):
            all_messages = load_obj(file)
            all_messages.reverse()
            for i in range(len(all_messages)):
                m = all_messages[i]
                if m["date"] <= min_datetime:
                    continue
                if str(m["peer_id"]["channel_id"]) + "-" + str(m["id"]) in labeled_samples:
                    continue

                print(str(i+1)+"/"+str(len(all_messages)) +" id:"+ str(m["id"]) +"-----"+ str(m["date"]) + "--------------")
                print("")
                print(m["message"])
                label = input("")
                if label == "##":
                    break # jump to the next channel
                elif label == "?" or label == "1" or label == "2":
                    pass # nothing
                elif label == "-1":
                    # back to the next message and relabel
                    pass
                else:
                    label = "0"

                f.write(str(label) + " " + str(m["peer_id"]["channel_id"]) + " " + str(m["id"]) + "\n")
