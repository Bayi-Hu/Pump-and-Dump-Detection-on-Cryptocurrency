import os
from PostInfoExtraction import Message, load_obj

if __name__ == '__main__':
    all_channels_urls = []
    for root, dirs, files in os.walk("./Data"):

        for file in files:
            if (file.startswith("@") and file.endswith(".pkl")):
                # all_messages = load_obj("@Crypto_Bitcoin_Pump##1198780255.pkl")
                all_messages = load_obj(file)
                channel_urls = []
                for m in all_messages:
                    try:
                        msg = Message(m["message"], m["date"])
                        if (len(msg.urls)) > 0:
                            channel_urls += msg.urls
                    except:
                        continue

                if len(channel_urls) > 0:
                    all_channels_urls += channel_urls
                    print(channel_urls)

    print("pause")

with open("invite_urls_onehop", "wb") as f:

    f.write(all_channels_urls_new)


