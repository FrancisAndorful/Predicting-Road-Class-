# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:45:22 2024

@author: 50409
"""
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
def points_keys():
    file_paths=['./GCN_Desseldorf_Data.csv','./GCN_HD_Data.csv','./GCN_Kalsruhe_Data.csv','./GCN_Munchen_Data.csv','GCN_Hamburg_Data.csv','GCN_Berlin_Data.csv']
    point_keys=[]
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = [row[4:-2] for row in reader]
            column_sums = [sum(0 if row[i] == '' else float(row[i]) for row in data[1:]) for i in range(len(data[1]))]
            
            columns_to_remove = [i for i, column_sum in enumerate(column_sums) if column_sum < 10]
            
            filtered_data = [[row[i] for i in range(len(row)) if i not in columns_to_remove] for row in data]
            point_keys.extend(filtered_data[0])
    
    unique_elements = sorted(set(point_keys))
    return unique_elements
'''
# 原列表
old_list = [['A', 'B', 'D', 'G'], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

# 表头列表
headers = ['A', 'B', 'D','V']

# 生成新的列表
new_list = []

# 将匹配的表头加入新的列表
new_list.append(headers)

# 找到匹配的表头在原列表中的索引
matching_indices = [old_list[0].index(header) if header in old_list[0] else -1 for header in headers]

# 从原列表中获取相应的元素，并填充0
for row in old_list[1:]:
    new_row = []
    for index in matching_indices:
        if index != -1:
            new_row.append(row[index])
        else:
            new_row.append(0)
    new_list.append(new_row)

# 输出新的列表
print(new_list)
'''
