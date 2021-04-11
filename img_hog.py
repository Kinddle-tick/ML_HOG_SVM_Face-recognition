from skimage.feature import hog
from skimage import io
from PIL import Image
import cv2
import os
import re
import json
import numpy as np
import time

The_orientations=8
The_pixels_per_cell=(30,30)
The_cells_per_block=(20, 20)
Test_num = 100
# hog(os.path.join(path_fake, files_fake[0]), orientations=The_orientations, pixels_per_cell=The_pixels_per_cell,
#     cells_per_block=The_cells_per_block, block_norm='L2-Hys', visualize=True)
flag=True
img=None

def hog_v(filename,clas):
    global flag,img
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    normalised_blocks = \
        hog(img, orientations=The_orientations, pixels_per_cell=The_pixels_per_cell,
            cells_per_block=The_cells_per_block, block_norm='L2-Hys',visualize=False)
    if flag==True:
        flag=False
        _,img = hog(img, orientations=The_orientations, pixels_per_cell=The_pixels_per_cell,
            cells_per_block=The_cells_per_block, block_norm='L2-Hys', visualize=True)
        io.imshow(img)
        io.imsave("{}.jpeg".format(The_cells_per_block),img)
        # io.show()
    # a = normalised_blocks.tolist()
    # a.append(clas)
    return np.hstack([normalised_blocks,clas])



t = time.time_ns()
path_fake = "/Users/kinddle/课程任务/人工智能基础/Project3/real_and_fake_face_training/training_fake" #文件夹目录
clas = 0  #0-fake 1-real,懒得写判断，自己标
files_fake = os.listdir(path_fake)
img_list_fake = np.empty(shape=[0, len(hog_v(os.path.join(path_fake, files_fake[0]), [0]))])
for filename in files_fake:

    img_list_fake = np.vstack([img_list_fake, hog_v(os.path.join(path_fake, filename), [0])])

np.save("fake_output", img_list_fake)
print(f"fake_output.npy saved, take time {(time.time_ns()-t)*1e-9}s")

t = time.time_ns()
path_test = "/Users/kinddle/课程任务/人工智能基础/Project3/real_and_fake_face_training/real_and_fake_face_testing" #文件夹目录
clas = 0  #0-fake 1-real,懒得写判断，自己标
files_test = os.listdir(path_test)
files_test =sorted(files_test,key=lambda x : int(re.findall("img(.*).jpg",x)[0]))
img_list_fake = np.empty(shape=[0, len(hog_v(os.path.join(path_test, files_test[0]), [0]))])
for filename in files_test:
    # print(filename)
    img_list_fake = np.vstack([img_list_fake, hog_v(os.path.join(path_test, filename), [0])])

np.save("testing_output", img_list_fake)
print(f"testing_output.npy saved, take time {(time.time_ns()-t)*1e-9}s")


t = time.time_ns()
path_real = "/Users/kinddle/课程任务/人工智能基础/Project3/real_and_fake_face_training/training_real" #文件夹目录
# clas = 1  #0-fake 1-real,懒得写判断，自己标
files_real = os.listdir(path_real)
img_list_real = np.empty(shape=[0, len(hog_v(os.path.join(path_real, files_real[0]), [0]))])
for filename in files_real:

    img_list_real = np.vstack([img_list_real, hog_v(os.path.join(path_real, filename), [1])])

np.save("real_output", img_list_real)
print(f"real_output.npy saved, take time {(time.time_ns()-t)*1e-9}s")



all_img=np.vstack([img_list_fake,img_list_real])
np.save("all_output",all_img)


# choose = np.random.randint(0,len(all_img),size=Test_num)
# test_img=all_img[choose]
# np.save("test_output",test_img)
#
# train_img = np.delete(all_img,choose,axis=0)
# np.save("train_output",train_img)

