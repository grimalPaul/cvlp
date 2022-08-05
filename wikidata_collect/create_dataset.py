import json
from datasets import load_from_disk, disable_caching

disable_caching()


def name_image(urls):
    file_names = []
    for url in urls:
        file_name = url.split('/')[-1]
        file_names.append(file_name)
    return file_names

def mapping_fct(item, data):
    # save url
    item['url_images'] = data[item['wikidata_id']]
    # save names of where the image is saved
    item['list_images'] = name_image(item['url_images']) 
    return item

if __name__ == '__main__':
    with open("data/wikimage.json", "r") as f:
        data = json.load(f)

    path_dataset = "data/wikimage"
    dataset = load_from_disk(path_dataset)

    keys = data.keys()
    new_dataset = dataset.filter(lambda x: x in keys, input_columns="wikidata_id")
    kwargs = {
        "data":data
    }
    new_dataset = new_dataset.map(mapping_fct,fn_kwargs=kwargs)
    path_output = "data/wikimage_with_multiple_image"
    new_dataset.save_to_disk(path_output)
