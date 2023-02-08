
# blob -> list names of files in folder
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

root = '/root/o2ac-ur/yolov5-pip/yolov5/dataset/synthetic_cooking'
img_files = glob.glob(root + "/images/*.png")
files = glob.glob(root + "/labelTxt/*.txt")

filenames = []

for f in files:
    # filenames.append(f.split('/')[-1].replace('.xml',''))
    filenames.append(f.split('/')[-1].replace('.txt', ''))

# print(filenames)

# df = pd.read_csv('/root/o2ac-ur/cooking_dataset/ImageSets/Main/full_database.txt', sep=',', header=None, names=["file", "data"])
# print(df)

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
train_size = 0.8

X = pd.DataFrame(filenames)
# X = df.drop(columns = ['data']).copy() #filename
# y = df['data'] #label

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, X, test_size=0.2, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

train_files = X_train[0].values.tolist()
test_files = X_test[0].values.tolist()
valid_files = X_valid[0].values.tolist()

# print(valid_files)

###### Create directories ######

train_dir = root + "/train"
test_dir = root + "/test"
valid_dir = root + "/valid"

for dir in ["train", "test", "valid"]:
    label_dir = Path(root).joinpath("labelTxt", dir)
    image_dir = Path(root).joinpath("images", dir)
    if not Path.exists(label_dir):
        Path.mkdir(label_dir)
    if not Path.exists(image_dir):
        Path.mkdir(image_dir)


def get_file(file, file_list):
    for f in file_list:
        if file in f:
            return f


###### Move files to directories #####
for f in train_files:
    in_label_file = Path(root).joinpath("labelTxt", f + ".txt")
    in_image_file = Path(root).joinpath("images", f + ".png")

    in_label_file.rename(Path(root).joinpath("labelTxt", "train", f + ".txt"))
    in_image_file.rename(Path(root).joinpath("images", "train", f + ".png"))

for f in test_files:
    in_label_file = Path(root).joinpath("labelTxt", f + ".txt")
    in_image_file = Path(root).joinpath("images", f + ".png")

    in_label_file.rename(Path(root).joinpath("labelTxt", "test", f + ".txt"))
    in_image_file.rename(Path(root).joinpath("images", "test", f + ".png"))

for f in valid_files:
    in_label_file = Path(root).joinpath("labelTxt", f + ".txt")
    in_image_file = Path(root).joinpath("images", f + ".png")

    in_label_file.rename(Path(root).joinpath("labelTxt", "valid", f + ".txt"))
    in_image_file.rename(Path(root).joinpath("images", "valid", f + ".png"))


# X_train.to_csv(r'/root/o2ac-ur/yolov5-pip/yolov5/dataset/synthetic_cooking/train.txt', header=None, index=None, sep='\t', mode='w')
# X_valid.to_csv(r'/root/o2ac-ur/yolov5-pip/yolov5/dataset/synthetic_cooking/test.txt', header=None, index=None, sep='\t', mode='w')
# X_test.to_csv(r'/root/o2ac-ur/yolov5-pip/yolov5/dataset/synthetic_cooking/valid.txt', header=None, index=None, sep='\t', mode='w')
