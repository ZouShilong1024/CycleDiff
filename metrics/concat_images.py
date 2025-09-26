import os
import cv2
import numpy as np
import shutil

gt_path = "/home/zsl/Data/afhqv2/gt_test_cat"
our_results_path = "/home/zsl/Data/afhqv2/translation_cat2dog_cat13_dog18_sample_from_eps_45"
save_path = "/home/zsl/Data/afhqv2/comparision"

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
else:
    shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

gt_images = [os.path.join(gt_path, file) for file in os.listdir(gt_path)]

for image in gt_images:
    img = cv2.imread(image)

    our_img = cv2.imread(os.path.join(our_results_path, os.path.basename(image)))

    out_img = np.concatenate((img, our_img), axis=1)

    cv2.imwrite(os.path.join(save_path, os.path.basename(image)), out_img)