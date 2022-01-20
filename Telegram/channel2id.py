import configparser
import json
import os
import pickle


from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

from telethon.tl.functions.messages import (GetHistoryRequest)
from telethon.tl.types import (
PeerChannel
)

api_id = 10732642
api_hash = 'aa62488becc0e9765efaea67811f5de5'
client = TelegramClient('anon', api_id, api_hash, proxy=("socks5", '127.0.0.1', 7890))
id2channel = {}
channel2id = {}


def save_obj(obj, name):
    with open("./Data/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

async def main():

    with open("./Data/PUMPOLYMP_public_channel", "r") as f:
        Channels = f.readlines()

    failed_channel = []
    for channel in Channels:
        entity = channel[13:-1]

        try:
            my_channel = await client.get_entity(entity)
            print(entity + ":" + str(my_channel.id))
        except:
            print("entity: " + entity + " can not be use!")
            failed_channel.append(entity)
            continue

        id2channel[my_channel.id] = entity
        channel2id[entity] = my_channel.id
    
    
with client:
    client.loop.run_until_complete(main())
    client.disconnect()

    print("pause")
