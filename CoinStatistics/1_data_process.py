import zipfile
import os

def unzip(src_file, dest_dir):
    """ungz zip file"""
    zf = zipfile.ZipFile(src_file)
    try:
        zf.extractall(path=dest_dir)
    except e:
        print(e)
    zf.close()

if __name__ == '__main__':

    dest_dir = "./data/unzip"
    file_list = []
    for root, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith(".zip"):

                unzip(os.path.join(root, file), dest_dir)

    print("pause")