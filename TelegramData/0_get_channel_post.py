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

def save_obj(obj, name):
    with open("./raw/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

async def main():

    with open("raw/PUMPOLYMP_public_channel2", "r") as f:
        Channels = f.readlines()

    failed_channel = []
    exist_channel = list(map(lambda y: y[:-4], os.listdir("raw")))

    for channel in Channels:

        offset_id = 0
        limit = 500
        all_messages = []
        total_messages = 0
        # total_count_limit = 0
        total_count_limit = 500000
        entity = channel[13:-1]

        try:
            my_channel = await client.get_entity(entity)
            if not my_channel.broadcast:
                print("entity: " + entity + " is not broad cast channel")
                continue

        except:
            print("entity: " + entity + " can not be use!")
            failed_channel.append(entity)
            continue

        if str(my_channel.id) in exist_channel:
            print("entity: " + entity + " has existed")
            continue

        while True:
            print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
            history = await client(GetHistoryRequest(
                peer=my_channel,
                offset_id=offset_id,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0
            ))
            if not history.messages:
                break
            messages = history.messages
            for message in messages:
                all_messages.append(message.to_dict())
            offset_id = messages[len(messages) - 1].id
            total_messages = len(all_messages)
            if total_count_limit != 0 and total_messages >= total_count_limit:
                break

        print("pause")
        save_obj(all_messages, str(my_channel.id))


with client:
    client.loop.run_until_complete(main())

# def load_obj(name):
#     with open("./raw/" + name +".pkl", "rb") as f:
#         return pickle.load(f)