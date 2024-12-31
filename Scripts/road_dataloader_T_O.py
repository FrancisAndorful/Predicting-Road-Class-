# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:58:27 2024

@author: Li
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ast
from points_keys import points_keys
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.label_map = {
            "living_street": 0,
            "residential": 1,
            "motorway_link": 2,
            "secondary_link": 3,
            "tertiary": 4,
            "unclassified": 5,
            "pedestrian": 6,
            "tertiary_link": 7,
            "track_grade2": 8,
            "footway": 9,
            "primary_link": 10,
            "track_grade5": 11,
            "secondary": 12,
            "busway": 13,
            "track": 14,
            "steps": 15,
            "primary": 16,
            "track_grade4": 17,
            "track_grade3": 18,
            "path": 19,
            "cycleway": 20,
            "trunk_link": 21,
            "bridleway": 22,
            "service": 23,
            #"highway": 24,
            "track_grade1": 24,
            "trunk": 25,
            "motorway": 26,
            "others":27
        }
        #self.frequency_vector = np.zeros(len(self.label_map))
        headers=points_keys()
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            
            old_list = [row[4:-2] for row in reader]
            file.seek(0)
            reader = csv.reader(file)
            ext=[row for row in reader]
            new_list = []
            new_list.append(headers)
            matching_indices =[old_list[0].index(header) if header in old_list[0] else -1 for header in headers]
            for row ,row_ext in zip(old_list[1:],ext[1:]):
                new_row = []
                for index in matching_indices:
                    if index != -1:
                        new_row.append(row[index])
                    else:
                        new_row.append(0)
                new_list.append(row_ext[0:4]+new_row+row_ext[-2:])
            
                        
            #next(reader) 
            reader=new_list[1:]
            for row in reader:
                
                label = self.label_map.get(row[2], 27)
                ##label = self.label_map[row[2]]  
                ##data = np.array(row[3:], dtype=float)
                data = [float(row[-2])]
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.long)
                
                fres=row[-1]
                fres = ast.literal_eval(fres)
                self.frequency_vector = np.zeros(len(self.label_map))
                for fre in fres:
                    #print(fre)
                    #label_index = self.label_map[fre.pop()]
                    label_index = self.label_map.get(fre.pop(), 27)
                    self.frequency_vector[label_index] = 1
                self.transform = transform
                
                data_con = np.concatenate((data, self.frequency_vector))
                
                self.data.append((data_con, label))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]

        if self.transform:
            data = self.transform(data)

        return data, label