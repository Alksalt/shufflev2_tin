import os, shutil

val_dir = "D:/datasets/tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
annotations = os.path.join(val_dir, "val_annotations.txt")

# create subfolders per class
with open(annotations) as f:
    for line in f:
        img, label = line.split("\t")[:2]
        label_dir = os.path.join(val_dir, "images_by_label", label)
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(images_dir, img)
        dst = os.path.join(label_dir, img)
        if os.path.exists(src):
            shutil.move(src, dst)
