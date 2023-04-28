import matplotlib.pyplot as plt
import numpy as np

f = open("/home/tanluuuuuuu/Desktop/luunvt/oriented_object_detection/mmrotate/work_dirs/oriented_rcnn_r50_fpn_1x_dota_le90_examine/check_xywha.txt", 'r')
# x = [i.strip().split("\t")[0] for i in f.readlines()]
# y = [i.strip().split("\t")[1] for i in f.readlines()]
# w = [i.strip().split("\t")[2] for i in f.readlines()]
# h = [i.strip().split("\t")[3] for i in f.readlines()]
lines = f.readline()
print(lines[1])

print(len(lines))
# print(len(x))
# print(len(y))
# print(len(w))
# print(len(h))
# plt.scatter(x, y)