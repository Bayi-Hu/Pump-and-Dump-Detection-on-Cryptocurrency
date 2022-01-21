import configparser
import json
import os.path
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

async def main():

    offset_id = 0
    limit = 100
    all_messages = []
    total_messages = 0
    total_count_limit = 0

    user_input_channel = input("enter entity(telegram URL or entity id):")

    if user_input_channel.isdigit():
        entity = PeerChannel(int(user_input_channel))
    else:
        entity = user_input_channel

    my_channel = await client.get_entity(entity)

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

    def save_obj(obj, name):
        with open("./Data/" + name + ".pkl", "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    save_obj(all_messages, user_input_channel + "##" + str(my_channel.id))

with client:
    client.loop.run_until_complete(main())

# save

def save_obj(obj, name):
    with open("./Data/" + name +".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# def load_obj(name):
#     with open("./Data/" + name +".pkl", "rb") as f:
#         return pickle.load(f)

