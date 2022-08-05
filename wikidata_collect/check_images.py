from faulthandler import disable
from datasets import load_from_disk, disable_caching
import os
import urllib
import urllib.request

disable_caching()

class Downloader(object):
    def __init__(self) -> None:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent',
                            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36 wikimageBot/1.0 (https://github.com/Paulgrim; pologri6800@gmail.com)')]
        urllib.request.install_opener(opener)
        self.count = 0
        self.name = "paulGrimal"
        self.change_name = {}

    def download_image(self,url, path, file_name):
        full_path = path + file_name
        status = urllib.request.urlretrieve(url, full_path)

    
def mappingImage(item, path, downloader:Downloader):
    list_url = item["url_images"]
    title = item["wikipedia_title"]
    new_list = list()
    count= 0
    for url in list_url:
        file_name = url.split('/')[-1]
        if not os.path.isfile(f"{path}{file_name}"):
            try:
                downloader.download_image(url=url,path=path, file_name=file_name)
                new_list.append(file_name)
            except urllib.error.HTTPError:
                print(f"pbm url : {url} \n {title}")
            except OSError:
                type_file = file_name.split('.')[-1]
                new_name = title + str(count) +"."+type_file
                count +=1
                downloader.download_image(url, path, new_name)
                new_list.append(new_name)
            except:
                print(f"other pbm occurs for url : {url} / {title}")
        else:
            new_list.append(file_name)
    item['list_images'] = new_list
    return item

if __name__ == '__main__':
    path_dataset = "data/wikimage_with_multiple_image"
    dataset = load_from_disk(path_dataset)

    pathImage = "data/Commons_wikimage/"
    downloader = Downloader()
    kwargs={
        "path":pathImage,
        "downloader":downloader
    }
    dataset = dataset.map(mappingImage, fn_kwargs=kwargs, num_proc=6)
    dataset.save_to_disk(path_dataset)
