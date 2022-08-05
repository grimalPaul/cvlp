from datasets import load_from_disk, disable_caching
import os
from tqdm import tqdm

disable_caching()
def remove_lines():
    path_dataset = "data/wikimage_with_multiple_image"
    dataset = load_from_disk(path_dataset)

    cpt = 0
    remove = []
    for i, item in enumerate(dataset):
        if len(item['list_images']) < 2:
            print(i)
            print(item['wikidata_id'])
            remove.append(i)
            cpt +=1

    print("lines to remove : ", cpt)
    if cpt >0:
        size = dataset.num_rows
        index = [i for i in range(size)]
        for i in remove:
            index.remove(i)
        dataset = dataset.select(index)
        dataset.save_to_disk(f"{path_dataset}2")


def check_images():
    path_dataset = "data/datasets/wikimage_no_filter"
    path_image = "data/Commons_wikimage/"
    dataset = load_from_disk(path_dataset)
    cpt = 0
    for index, list_images in enumerate(tqdm(dataset['list_images'])):
        for image in list_images:
            if not os.path.isfile(f"{path_image}{image}"):
                print(f"pbm{index} : {image}")
                cpt +=1
    print(cpt)

if __name__=='__main__':
    check_images()

"""
remove_id = [
    "Q6014271",
    "Q3290473",
    "Q87306",
    "Q7961593",
    "Q22981838"
]
"""