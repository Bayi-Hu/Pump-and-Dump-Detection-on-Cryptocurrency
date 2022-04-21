import os
import pickle
from urlextract import URLExtract

def load_obj(file):
    with open("./raw/" + file, "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    all_channels_urls = []
    for root, dirs, files in os.walk("raw"):

        for file in files:
            if file.endswith(".pkl"):
                # all_messages = load_obj("@Crypto_Bitcoin_Pump##1198780255.pkl")
                all_messages = load_obj(file)
                channel_urls = []
                for m in all_messages:

                    if m["from_id"] != None or m["_"] != "Message":
                        continue
                    try:
                        extractor = URLExtract()
                        urls = extractor.find_urls(m["message"])
                        if len(urls) == 0:
                            continue
                        for url in urls:
                            if "t.me" in url:
                                all_channels_urls.append(url)
                    except:
                        continue

    print("pause")


