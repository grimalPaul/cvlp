import json
import urllib.request
from tqdm import tqdm
import multiprocessing as mp
import argparse
import urllib

class Downloader(object):
    def __init__(self) -> None:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent',
                            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36 wikimageBot/1.0 (https://github.com/Paulgrim; grimal-paul@outlook.fr)')]
        urllib.request.install_opener(opener)
        self.count = 0
        self.name = "paulGrimal"
        self.change_name = {}

    def download_image(self,url, path, file_name):
        full_path = path + file_name
        try:
            status = urllib.request.urlretrieve(url, full_path)
        except urllib.error.HTTPError:
                print("error http")
    
    def save_image(self,urls, path, step):
        file_names = []
        for url in urls:
            file_name = url.split('/')[-1]
            try:
                self.download_image(url, path, file_name)
                file_names.append(file_name)
            except OSError:
                type_file = file_name.split('.')[-1]
                new_name = self.name + str(self.count) + type_file
                self.change_name[new_name] = file_name
                print(f"name error {step} new {new_name}, old : {file_name}")
                self.count +=1
                self.download_image(url, path, new_name)
                file_names.append(new_name)
            
            except:
                print(f'step : {step}\n file_name : {file_name}')
        return file_names

    def save_log(self, path):
        with open(path, 'w') as f:
            json.dump(self.change_name, f)

if __name__ == '__main__':
    # en(data.keys) = 43747
    # / 4
    # 0 : 10936
    # 10936 : 21873
    # 21873 : 32810
    # 32810 : 43747
    # python download_image.py --log_path=log1 --start=0 --end=10936 --ckpt=4000
    # python download_image.py --log_path=log2 --start=10936 --end=21873 --ckpt=6055
    # python download_image.py --log_path=log3 --start=21873 --end=32810 --ckpt=7018
    # python download_image.py --log_path=log4 --start=32810 --end=43747 --ckpt=6767
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True, default=43747)
    parser.add_argument('--ckpt', type=int, required=False, default=0)
    args = parser.parse_args()

    with open("data/wikimage.json", "r") as f:
        data = json.load(f)
    path='data/Commons_wikimage/'
    downloader = Downloader()
    keys = list(data.keys())
    if args.ckpt != 0:
        start = args.ckpt + args.start
    else:
        start = args.start
    keys = keys[start:args.end]
    for step, k in tqdm(enumerate(keys)):
        urls = data[k]
        downloader.save_image(urls = urls,path = path, step = step)
    downloader.save_log(args.log_path)

