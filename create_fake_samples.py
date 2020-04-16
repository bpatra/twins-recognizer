from PIL import Image, ImageDraw
import random
import numpy as np

source="./data/sets/test/J/2_DSC00191.JPG"
dest="./data/sets_faked/test/J/2_DSC00191.JPG"

def create_faked_image(is_rectangle, source_img, dest_img):
    img = Image.open(source_img)
    w, h= img.size
    w2 = random.randrange(10, w / 2)
    h2 = random.randrange(10, w / 2)
    x=random.randrange(0, w / 2)
    y=random.randrange(0, h/2)
    bbox =[x, y, x+w2, y + h2]
    draw = ImageDraw.Draw(img)
    if is_rectangle:
        draw.rectangle(bbox, fill=tuple(np.random.randint(256, size=3)))
    else:
        draw.ellipse(bbox, fill=tuple(np.random.randint(256, size=3)))
    img.save(dest_img, quality=100)

json_pattern = "./data/target_vott/*.json"
file_count = len(glob.glob(json_pattern))

create_faked_image(True, source, dest)