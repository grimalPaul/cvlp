from datasets import load_from_disk, disable_caching
from PIL import Image
import cv2
from cairosvg import svg2png
from io import BytesIO
import numpy as np
import json
import PIL
from tqdm import tqdm

disable_caching()

#PIL.Image.MAX_IMAGE_PIXELS = None

def test_format(name):
    if name[-4:] in ['0jpg','0JPG','1jpg','2jpg','3jpg','4jpg','0tif','5jpg','0png','6jpg','7jpg','8jpg','1tif']:
        return True
    return False

def check_format(dataset):
    format = {}
    file_ex = {}
    cpt = 0
    for images in dataset['list_images']:
        for im in images:
            cpt +=1
            if im[-4:] in format:
                format[im[-4:]] += 1
                if test_format(im):
                    file_ex[im[-4:]].append(im)
            else:
                format[im[-4:]] = 1
                if test_format(im):
                    file_ex[im[-4:]] = []
                    file_ex[im[-4:]].append(im)
    print(format, cpt)
    #to_change_name("data/file2change_name.json",file_ex)
    #test_open_file(file_ex=file_ex, path = "data/Commons_wikimage")

def test_open_file(path, file_ex):
    for type, im in file_ex.items():
        im = f'{path}/{im}'
        if type.lower() == ".gif":
            img = np.array(Image.open(im).convert('RGB'))
        elif type.lower() == ".svg":
            png = svg2png(file_obj=open(im,'r'))
            img = np.array(Image.open(BytesIO(png)).convert('RGB'))
        elif type ==".pdf":
            pass
        elif type == ".xcf":
            pass
        elif type == "djvu":
            pass
        else:
            Image.open(im)
            img = cv2.imread(im)

def test_open_file_dataset(dataset,path):
    for list_image in tqdm(dataset['list_images']):
        for image in list_image:
            try: 
                im = f'{path}/{image}'
                type = im[-4:]
                if type.lower() == ".gif":
                    img = np.array(Image.open(im).convert('RGB'))
                elif type.lower() == ".svg":
                    png = svg2png(file_obj=open(im,'r'))
                    img = np.array(Image.open(BytesIO(png)).convert('RGB'))
                else:
                    Image.open(im)
                    img = cv2.imread(im)
            except:
                print(image)


change_name = []
def map_delete_pdf_xcf_djvu(item):
    list_image = item['list_images']
    new_list = []
    global change_name
    for image in list_image:
        if image[-4:] not in ['.pdf', '.xcf', 'djvu']:
            if image[-4:] in ['0jpg','0JPG','1jpg','2jpg','3jpg','4jpg','0tif','5jpg','0png','6jpg','7jpg','8jpg','1tif']:
                new_name = image[:-3] + "." +image[-3:]
                change_name.append(f'mv {image} {new_name}')
                new_list.append(new_name)
            else :
                new_list.append(image)
    item['list_images'] = new_list
    return item

def to_change_name(file_path, file):
    with open(file_path, 'w') as f:
        json.dump(file, f)

if __name__ == '__main__':
    path_dataset = 'data/datasets/wikimage_no_filter'
    dataset = load_from_disk(path_dataset)
    # check_format(dataset)
    test_open_file_dataset(dataset, path="data/Commons_wikimage")
    # dataset = dataset.map(map_delete_pdf_xcf_djvu)
    # file = {}
    # file['change_name']= change_name
    # to_change_name(file_path = "data/command", file =change_name)
    # dataset.save_to_disk(path_dataset)