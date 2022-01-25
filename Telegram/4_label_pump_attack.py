import numpy as np
import pandas as pd
import pickle

def load_obj(file):
    with open("./PreSelection/" + file, "rb") as f:
        return pickle.load(f)

pred_pump_sample = pd.read_csv("./Labeled/pred_pump_annouce.csv", sep="\t")
print("pause")

channels = set(pred_pump_sample.channel_id.values)
# select from pre-selection samples

# for cid in channels:
#     all_messages = load_obj(str(cid)+".pkl")

all_messages = load_obj("1553551852.pkl")


