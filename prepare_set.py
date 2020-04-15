import glob
import math
import random
import shutil

source_dir_L = "./data/cropped/L/"
source_dir_J = "./data/cropped/J/"

target_train = "./data/sets/train/"
target_validation = "./data/sets/validation/"
target_test = "./data/sets/test/"

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

for i in range(0, kept_length - val_size - test_size):
    shutil.copy(all_j_files[i], target_train + "J/")
    shutil.copy(all_l_files[i], target_train + "L/")

for i in range(kept_length - val_size - test_size, kept_length - test_size):
    shutil.copy(all_j_files[i], target_validation + "J/")
    shutil.copy(all_l_files[i], target_validation + "L/")

for i in range(kept_length - test_size, kept_length):
    shutil.copy(all_j_files[i], target_test + "J/")
    shutil.copy(all_l_files[i], target_test + "L/")