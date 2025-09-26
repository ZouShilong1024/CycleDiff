import os
import cv2
import shutil

path = "/home/zsl/Data/afhqv2/test/cat"
save_path = "/home/zsl/Data/afhqv2/gt_test_cat"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
else:
    shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

images_list = [os.path.join(path, f) for f in os.listdir(path)]

for image in images_list:
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256), )
    cv2.imwrite(os.path.join(save_path, os.path.basename(image)), img)