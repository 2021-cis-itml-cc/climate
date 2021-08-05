#!/usr/bin/env python
# coding: utf-8

# GRU
import torch
import matplotlib.pyplot as plt
import PyEMD
import numpy as np

import file_walker
import publicMethod

SITE_CODE = "722860"
paths = file_walker.file_walker().main(SITE_CODE)
print("data count:", len(paths))

print("One whole year tempreture form %s "%SITE_CODE)
tem_avg, tem_min, tem_max = {}, {}, {}
for path in paths:
    year = path.split('/')[-1].split("-")[-1].split(".")[0]
    tem_avg[year], tem_min[year], tem_max[year] = publicMethod.open_file(path)

plt.plot(list(tem_avg.values())[-1])
plt.xlabel("data from %s"%SITE_CODE)
plt.ylabel("Temperature/F")
plt.title("One whole year tempreture form %s "%SITE_CODE)
plt.show()

print("tem_avg_%s.jpg"%SITE_CODE)
tem_avg_list_i = list(tem_avg.values())
tem_avg_list = []
for i in tem_avg_list_i:
    tem_avg_list.extend(i)
plt.plot(tem_avg_list)
plt.gcf().set_size_inches(90, 10)
plt.xlabel("Days from %s"%SITE_CODE)
plt.ylabel("Temperature/F")
plt.title("All Temperatures of Station %s 1929-2021"%SITE_CODE)
plt.show()
plt.savefig("./fig/tem_avg_%s.jpg"%SITE_CODE)

print("imfs_%s.jpg"%SITE_CODE)
tem_avg_array = np.array(tem_avg_list)
emd_obj = PyEMD.EMD()
imfs = emd_obj.emd(tem_avg_array)
plt.plot(imfs.T)
plt.show()
plt.savefig("./fig/imfs_%s.jpg"%SITE_CODE)


print("imfs_subplot_%s.jpg"%SITE_CODE)
fig, axes = plt.subplots(len(imfs), 1)
_ = [axes[i].plot(imfs[i]) for i in range(len(imfs))]
plt.gcf().set_size_inches(70, 100)
plt.savefig("./fig/imfs_subplot_%s.jpg"%SITE_CODE)


