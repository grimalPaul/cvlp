from datasets import load_from_disk, disable_caching
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import PIL

disable_caching()

list_images  = [
    "Status%20iucn3.1%20VU.svg",
    "Status%20iucn3.1%20LC.svg",
    "PTSans.svg",
"Trinity%20College%20Oxford%20Coat%20Of%20Arms.svg,"
    "Bertrand2-figure.svg",
    "Bertrand3-figure.svg",
    "Blessed%20Osanna%20Andreasi%20-%20Unknown%20artist%2016th%20century.jpg",
    "Las%20Meninas%2C%20by%20Diego%20Vel%C3%A1zquez%2C%20from%20Prado%20in%20Google%20Earth.jpg",
    "LRO%20Tycho%20Central%20Peak.jpg",
    "Erinnyis%20crameri%20MHNT%20CUT%202010%200%20524%20Valle%20de%20Cochabamba%20Bolivia%20-%20male.jpg",
    "Rudolf%20Ulrich%20Kroenlein.tif",
    "Harold%20Jones%2C%20drummer.png",
    "Michelsonmorley-boxplot.svg",
    "LRO%20Tycho%20Central%20Peak.jpg",
    "Status%20iucn3.1%20CR.svg",
    "Status%20iucn3.1%20EW.svg",
    "Latin%20alphabet%20Jj.svg",
    "Status%20iucn3.1%20EN.svg",
    "Sandro%20Botticelli%20-%20La%20nascita%20di%20Venere%20-%20Google%20Art%20Project.jpg",
    "Pieter%20Bruegel%20the%20Elder%20-%20The%20Tower%20of%20Babel%20%28Vienna%29%20-%20Google%20Art%20Project%20-%20edited.jpg",
    "II%20Powstanie%20Pruskie.svg",
    "Trinity%20College%20Oxford%20Coat%20Of%20Arms.svg"
    "PTSans.svg",
    "PTSerif.svg",
    "Tevenphage.svg",
    "Clevis.svg",
    "P-500%20bazalt%20sketch.svg",
    "Faravahar-BW.svg",
    "El%20Tres%20de%20Mayo%2C%20by%20Francisco%20de%20Goya%2C%20from%20Prado%20in%20Google%20Earth.jpg",
    "Tottenham%20Outrage%20in%20The%20Illustrated%20London%20News%2C%2030%20January%201909%20%28retouched%29.jpg",
    "Las%20Meninas%2C%20by%20Diego%20Vel%C3%A1zquez%2C%20from%20Prado%20in%20Google%20Earth.jpg",
    "Helgoland%20Insel%20D%C3%BCne%202190-Pano.jpg",
    "1%20songzanlin%20monastery%20yunnan%202018.jpg",
    "Sesshu%20-%20Haboku-Sansui%20-%20complete.jpg",
    "Kloster%20Metten%20Panorama.jpg",
    "Status%20none%20DD.svg",
    "Calumet%2C%20Michigan%20panorama%20c1900.jpg",
    "Word%20Tamil.svg",
    "Hebrew%20alefbet%20vector.svg",
    "Jan%20van%20Eyck%20-%20Lucca%20Madonna%20-%20Google%20Art%20Project.jpg",
    "THX%20waterfall.svg",
    "Saint%20George%20and%20the%20princess%20by%20Pisanello%20-%20Pellegrini%20Chapel%20-%20Sant%27Anastasia%20-%20Verona%202016%20and%20corrections%20%28perspective%2C%20lights%2C%20definition%20by%20Paolo%20Villa%202019%29.jpg",
    "Familie%20Ceratopsidae.svg",
    "Saltwater%20Limpet%20Diagram-en.svg",
    "Kyoto%20protocol%20parties%20and%202012-2020%20commitments.svg",
    "Japanese%20Katakana%20ZA.svg",
    "Australian%20troops%20disembarking%20at%20Alexandria%20after%20the%20evacuation%20of%20Greece%20-%20Ivor%20Hele.jpg",
    "Go-shichi%20no%20kiri%20crest%202.svg",
    "Hanzi%20%28simplified%29.svg",
    "Great%20Lakes%20Lake%20Michigan.png",
    "Escudo%20de%20Librilla.svg",
    "Server%20rack%20rail%20dimensions.svg",
    "Tevenphage.svg",
    "Mandelbrot20210909%20ABC02%2065535x65535.png",
    "Kolakoski%20sequence%20spiral.svg",
    "Result%20of%20referendum%20of%20Osaka%20Metropolis%20plan%20in%2020150517.svg",
    "Conversion%20of%20Paul%20%28Bruegel%29.jpg",
    "Status%20iucn3.1%20NT.svg",
]

def map_remove(item):
    images = item['list_images']
    new_images = []
    for image in images:
        if image not in list_images:
            new_images.append(image)
    item['list_images'] = new_images
    return item

if __name__ == '__main__':
    path_dataset = "data/datasets/wikimage_no_filter"
    path_image = "data/Commons_wikimage/"
    dataset = load_from_disk(path_dataset)
    dataset =  dataset.map(map_remove)
    dataset = dataset.filter(lambda x: len(x) >= 2, input_columns='list_images')
    print(dataset)
    dataset.save_to_disk(path_dataset)