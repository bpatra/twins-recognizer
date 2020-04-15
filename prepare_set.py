import glob
import math
import random
import shutil
import os
import numpy as np
from PIL import Image, ImageDraw

def get_folder_name(add_fake, folder):
    folder_name="sets_faked" if add_fake else "sets"
    if folder == "train":   
        return f"./data/{folder_name}/train/"
    elif folder =="validation":
        return f"./data/{folder_name}/validation/"
    elif folder =="test":
        return f"./data/{folder_name}/test/"
    else:
        raise Error("folder kind not supported")

def create_faked_image(is_rectangle, source_img, dest_img):
    img = Image.open(source_img)
    w, h= img.size
    w2 = random.randrange(10, math.floor(w / 2))
    h2 = random.randrange(10, math.floor(h / 2))
    x=random.randrange(0, math.floor(w / 2))
    y=random.randrange(0, math.floor(h / 2))
    bbox =[x, y, x+w2, y + h2]
    draw = ImageDraw.Draw(img)
    if is_rectangle:
        draw.rectangle(bbox, fill=tuple(np.random.randint(256, size=3)))
    else:
        draw.ellipse(bbox, fill=tuple(np.random.randint(256, size=3)))
    img.save(dest_img, quality=100)

def process_files(add_fake):
    source_dir_L = "./data/cropped/L/"
    source_dir_J = "./data/cropped/J/"

    val_size=150
    test_size=150

    types = ('*.jpg', '*.JPG') # the tuple of file typesfiles_grabbed = []
    all_l_files=[]
    all_j_files=[]
    for type in types:
        all_l_files.extend(glob.glob(source_dir_L+type))
        all_j_files.extend(glob.glob(source_dir_J+type))

    print(f"found {len(all_j_files)} J files")
    print(f"found {len(all_l_files)} L files")

    kept_length=min(len(all_j_files), len(all_l_files))
    kept_j = random.shuffle(all_j_files[0:kept_length])
    kept_l = random.shuffle(all_j_files[0:kept_length])

    list = [("train",0, kept_length - val_size - test_size), 
            ("validation",kept_length - val_size - test_size, kept_length - test_size ),
            ("test",kept_length - test_size, kept_length )]

    for tuple in list:
        folder_name = tuple[0]
        start = tuple[1]
        end = tuple[2]
        print(f"processing files for {folder_name}")
        for i in range(0, kept_length - val_size - test_size):
            print(f"{i}/{kept_length - val_size - test_size} done!")
            normal_folder = get_folder_name(False, folder_name)
            shutil.copy(all_j_files[i], normal_folder + "J/")
            shutil.copy(all_l_files[i], normal_folder + "L/")
            if add_fake: # Create fake sets with shapes
                fake_folder = get_folder_name(True, folder_name)
                create_faked_image(True, all_j_files[i], fake_folder + "J/" + os.path.basename(all_j_files[i]))
                create_faked_image(False, all_l_files[i], fake_folder + "L/" + os.path.basename(all_l_files[i]))

process_files(True)