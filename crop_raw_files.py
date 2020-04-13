import glob
import os
import json
import pprint
import math
import urllib

from PIL import Image

def process():
    json_pattern = "./data/target_vott/*.json"
    file_count = len(glob.glob(json_pattern))
    print(f"Found {file_count} files in data directory!")

    counter = 0
    for file in glob.glob(json_pattern):
        counter += 1
        print(f"Processing file {file} - {counter}/{file_count}...")
        with open(file) as f:
            data = json.load(f)   
            image_file = data['asset']['path']
            # url encoded file paths...
            image_file2 = image_file[len("file:"):]
            image_file_decoded = urllib.parse.unquote(image_file2)
        
        if os.path.isfile(image_file_decoded):
            crop_file(image_file_decoded, data)
        else:
            raise FileNotFoundError(f"Cannot find file {image_file_decoded}")
        print(f"... {file} Ok!");

# Extract image samples and save to output dir
def crop_file(image_file, data):
    print(f"Cropping {image_file} objects on image")

    for region in data["regions"]:
        tag = region["tags"][0]
        if tag == "J" or tag == "L":
            basename = os.path.basename(image_file)
            result_file_path = f"./data/cropped/{tag}/{basename}"

            points = region["points"]
            img = Image.open(image_file)
            x_min = math.floor(points[0]["x"])
            y_min = math.floor(points[0]["y"])
            x_max = math.floor(points[2]["x"])
            y_max = math.floor(points[3]["y"])

            if x_max > x_min + 10 and y_max > y_min + 10:
                box = (x_min, y_min, x_max, y_max)
                print(f"Cropping size {img.size} with {box}")
                im = img.crop(box)
                im.save(result_file_path)
            else:
                print("ERROR: bounding box too small")
        else:
            print("ERROR: unrecognized tag")

if __name__ == '__main__':
    print("\n------------------------------------")
    print("----------- Cropp VoTT files ----------")
    print("------------------------------------\n")
    process()
