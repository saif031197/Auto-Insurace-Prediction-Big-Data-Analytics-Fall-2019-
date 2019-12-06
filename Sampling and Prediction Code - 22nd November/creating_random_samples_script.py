# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:08:07 2019

@author: sharm
"""

import pandas as pd
import numpy as np
import math

og_data = pd.read_csv(r"C:\Users\sharm\Desktop\Books\Semester 3\Big Data For Competative Advances\Big Data Project\Clean_Dataset_17nov.csv")

def compress_dataFrame(first):
    for row in first:
        if row == "Policy_Installment_Term":
            first[row] = first[row].mean()
        elif row == "Vehicle_Make_Year":
            first["Vehicle_Make_Year_max"] = first[row].max()
            first["Vehicle_Make_Year_min"] = first[row].min()
            first[row] = first[row].median()
        elif row == "Vehicle_Symbol":
            first["Vehicle_Symbol_max"] = first[row].max()
            first["Vehicle_Symbol_min"] = first[row].min()
            first[row] = first[row].median()    
        else:
            first[row] = first[row].sum()
            
    first_new = first[:1]
    return first_new

initial_sample = 1000

for i in range(12):
    loss_data = og_data[og_data['Loss_Amount'] > 0]
    noloss_data = og_data[og_data['Loss_Amount'] == 0]
    
    t_noloss_data = noloss_data.sample(frac=1, random_state=i).reset_index(drop=True)
    t_loss_data = loss_data.sample(frac=1, random_state=i).reset_index(drop=True)
    
    #t_loss_data = t_loss_data.sort_values(by=['Loss_Amount'])
    
    m = math.ceil(og_data.shape[0]/initial_sample)
    total_noloss = math.ceil(noloss_data.shape[0]/m)
    total_loss = initial_sample - total_noloss
    
    for p in range(11):
        noloss_data = t_noloss_data.sample(frac=1, random_state=p).reset_index(drop=True)
        loss_data = t_loss_data.sample(frac=1, random_state=p).reset_index(drop=True)
        
        '''if i in (3,4,5):
          print("Sort Descending")  
          loss_data = loss_data.sort_values(by=['Loss_Amount'],ascending=False)
          
        if i in (6,7,8):
          print("Sort Ascending") 
          loss_data = loss_data.sort_values(by=['Loss_Amount'],ascending=True)'''  
        
        if p == 10:
            p = 19
        
        loss_count = math.ceil((initial_sample*(p+1)/100))
        noloss_count = initial_sample - loss_count
        
        print("Loss Count =",loss_count)
        print("No loss count = ", noloss_count)
        
        chunked_dataframes = []
        
        while len(noloss_data) > noloss_count and len(loss_data) > loss_count:
            top = noloss_data[:noloss_count]
            top = top.append(loss_data[:loss_count])
            chunked_dataframes.append(top)
            noloss_data = noloss_data[noloss_count:]
            loss_data = loss_data[loss_count:]
        else:
            top = (noloss_data)
            top = top.append(loss_data)
            chunked_dataframes.append(top) 
                
        print("Length of Chuncked df = ", len(chunked_dataframes)) 
        print(chunked_dataframes[0].shape)
        print()
        
        merged_dataframes = []
        for dataframe in chunked_dataframes:
            merged_dataframes.append(compress_dataFrame(dataframe))
        
        sampled_dataframe = pd.concat([df for df in merged_dataframes])
        if i == 0:
            final_dataframe = sampled_dataframe
        else:
            final_dataframe = final_dataframe.append(sampled_dataframe)
        
    initial_sample+=2000
    
    if initial_sample == 7000:
        initial_sample = 1000
    
    
print("Final Dataset = ",final_dataframe.shape)
final_dataframe.to_csv(r"C:\Users\sharm\Desktop\Books\Semester 3\Big Data For Competative Advances\Big Data Project\Feature_Engineered_Dataset_Nov17.csv", index=False)
