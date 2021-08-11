from PIL import Image
import skimage.io as io
import json
import argparse
import tqdm

def load_file(json_file):
    r = None
    with open(json_file, "r") as f:
        r = json.load(f)
    return r

def save_img(dico, data_dir):
    if False :
        id = dico["id"]
        url = dico["url"]
        im = Image.fromarray(io.imread(url))
        im.save(f"{data_dir}/{id}.png")

def dl_pairs(pairs, data_dir):
    for p in tqdm.tqdm(pairs):
        for k, img in p.items():
            if isinstance(img, dict):
                save_img(img, data_dir)

def dl_singles(singles, data_dir):
    for img in tqdm.tqdm(singles):
        if isinstance(img, dict):
            save_img(img, data_dir)

if __name__ == "__mai__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs")
    parser.add_argument("--singles")
    parser.add_argument("--target")
    args = parser.parse_args()

    method = None
    input = None
    message = None
    if args.pairs is not None:
        message = "Donwload pairs"
        method = dl_pairs
        input = args.pairs       
    elif args.singles is not None:
        message = "Donwload singles"
        method = dl_singles
        input = args.singles
    
    print(message)
    method(load_file(input), args.target)

if __name__ == "__main__":
    dl_pairs()