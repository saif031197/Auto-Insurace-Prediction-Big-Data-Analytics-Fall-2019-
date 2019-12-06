# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
df=pd.read_csv("training_data.csv")
df_backup=pd.read_csv("training_data_shortened.csv")
df_final=pd.DataFrame()
df_final.insert(0,'Make',np.nan)
df_final.insert(1,'Model',np.nan)
df_final[['Make','Model']] = df['Vehicle_Make_Description'].str.split(' ',1).tolist()

#CHECKPOINT


df=df[df.Loss_Amount != 0]
df['Driver_Total'].value_counts()


import matplotlib.pyplot as plt
ax = df[['Driver_Minimum_Age',]].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("V", fontsize=12)
plt.show()
#df_final = df_final.drop('Make', 1).join(df_final.Make.str.get_dummies())
#df_final = df_final.drop('Model', 1).join(df_final.Model.str.get_dummies())
df_final=df_final.join(df.Vehicle_Usage.str.get_dummies().add_prefix('Vehicle_Usage_'))
df_final=df_final.join(df['Vehicle_Symbol'])
df_final=df_final.join(df['Vehicle_Make_Year'])
df_final=df_final.join(df['Vehicle_Collision_Coverage_Deductible'])
#df_final=df_final.join(df['Vehicle_Number_Of_Drivers_Assigned'])
df_final=df_final.join(df['Vehicle_Miles_To_Work'])
df_final=df_final.join(df.Vehicle_Performance.str.get_dummies().add_prefix('Vehicle_Performance_'))
df_final=df_final.join(df.Vehicle_Anti_Theft_Device.str.get_dummies().add_prefix('Vehicle_Anti_Theft_Device_'))
df_final=df_final.join(df.Vehicle_Passive_Restraint.str.get_dummies().add_prefix('Vehicle_Passive_restraint_'))
#df_final=df_final.join(df['Vehicle_Age_In_Years'])
df_final=df_final.join(df['Policy_Installment_Term'])
df_final=df_final.join(df['Vehicle_Med_Pay_Limit'])
df_final=df_final.join(df['Vehicle_Physical_Damage_Limit'])
df_final=df_final.join(df['Vehicle_Comprehensive_Coverage_Limit'])
df_final=df_final.join(df['Vehicle_New_Cost_Amount'])
df_final=df_final.join(df['Driver_Total'])
#df_final=df_final.join(df['Driver_Total_Male'])
df_final=df_final.join(df['Driver_Total_Female'])
df_final=df_final.join(df['Driver_Total_Single'])
#df_final=df_final.join(df['Driver_Total_Married'])
#df_final=df_final.join(df['Driver_Total_Related_To_Insured_Self'])
#df_final=df_final.join(df['Driver_Total_Related_To_Insured_Spouse'])
df_final=df_final.join(df['Driver_Total_Related_To_Insured_Child'])
df_final=df_final.join(df['Driver_Total_Licensed_In_State'])
#df_final=df_final.join(df['Driver_Minimum_Age'])
#df_final=df_final.join(df['Driver_Maximum_Age'])
df_final=df_final.join(df['Driver_Total_Teenager_Age_15_19'])
df_final=df_final.join(df['Driver_Total_College_Ages_20_23'])
df_final=df_final.join(df['Driver_Total_Young_Adult_Ages_24_29'])
df_final=df_final.join(df['Driver_Total_Low_Middle_Adult_Ages_30_39'])
df_final=df_final.join(df['Driver_Total_Middle_Adult_Ages_40_49'])
#df_final=df_final.join(df['Driver_Total_Adult_Ages_50_64'])
#df_final=df_final.join(df['Driver_Total_Senior_Ages_65_69'])
#df_final=df_final.join(df['Driver_Total_Upper_Senior_Ages_70_plus'])



df_final=df_final.join(df.Vehicle_Youthful_Driver_Indicator.str.get_dummies().add_prefix('Vehicle_Youthful_Driver_Indicator_'))
df_final=df_final.join(df.Vehicle_Youthful_Driver_Training_Code.str.get_dummies().add_prefix('Vehicle_Youthful_Driver_Training_Code_'))
df_final=df_final.join(df.Vehicle_Youthful_Good_Student_Code.str.get_dummies().add_prefix('Vehicle_Youthful_Good_Student_Code_'))

df_final=df_final.join(df['Vehicle_Driver_Points'])

df_final=df_final.join(df.Vehicle_Safe_Driver_Discount_Indicator.str.get_dummies().add_prefix('Vehicle_Safe_Driver_Discount_Indicator'))

''' EXTRA ONE HOT ENCODE
'''
df_final=df_final.join(df.Policy_Company.str.get_dummies().add_prefix('Policy_Company'))
df_final=df_final.join(df.EEA_Multi_Auto_Policies_Indicator.str.get_dummies().add_prefix('EEA_Multi_Auto_Policies_Indicator'))
df_final=df_final.join(df.EEA_Liability_Coverage_Only_Indicator.str.get_dummies().add_prefix('EEA_Liability_Coverage_Only_Indicator'))
df_final=df_final.join(df.EEA_Agency_Type.str.get_dummies().add_prefix('EEA_Agency_Type'))
df_final=df_final.join(df.EEA_Packaged_Policy_Indicator.str.get_dummies().add_prefix('EEA_Packaged_Policy_Indicator'))
df_final=df_final.join(df.EEA_Full_Coverage_Indicator.str.get_dummies().add_prefix('EEA_Full_Coverage_Indicator'))

df_final=df_final.join(df['Annual_Premium'])
df_final=df_final.join(df['Loss_Amount'])



df_final.to_csv('final_dataset_3.csv')


for x in range(1,601):
    try:
        df=pd.read_csv('test_portfolio_'+str(x)+'.csv')
        df_final=pd.DataFrame()
        df_final.insert(0,'Make',np.nan)
        df_final.insert(1,'Model',np.nan)
        df_final[['Make','Model']] = df['Vehicle_Make_Description'].str.split(' ',1).tolist()
        df_final=df_final.join(df.Vehicle_Usage.str.get_dummies().add_prefix('Vehicle_Usage_'))
        df_final=df_final.join(df['Vehicle_Symbol'])
        df_final=df_final.join(df['Vehicle_Number_Of_Drivers_Assigned'])
        df_final=df_final.join(df.Vehicle_Performance.str.get_dummies().add_prefix('Vehicle_Performance_'))
        df_final=df_final.join(df['Vehicle_Miles_To_Work'])
        df_final=df_final.join(df.Vehicle_Anti_Theft_Device.str.get_dummies().add_prefix('Vehicle_Anti_Theft_Device_'))
        df_final=df_final.join(df.Vehicle_Passive_Restraint.str.get_dummies().add_prefix('Vehicle_Passive_restraint_'))
        df_final=df_final.join(df['Vehicle_Age_In_Years'])
        df_final=df_final.join(df['Driver_Total'])
        df_final=df_final.join(df['Driver_Total_Male'])
        df_final=df_final.join(df['Driver_Total_Female'])
        df_final=df_final.join(df['Driver_Total_Single'])
        df_final=df_final.join(df['Driver_Total_Married'])
        df_final=df_final.join(df['Driver_Total_Related_To_Insured_Self'])
        df_final=df_final.join(df['Driver_Total_Related_To_Insured_Spouse'])
        df_final=df_final.join(df['Driver_Total_Related_To_Insured_Child'])
        df_final=df_final.join(df['Driver_Total_Licensed_In_State'])
        df_final=df_final.join(df['Driver_Minimum_Age'])
        df_final=df_final.join(df['Driver_Maximum_Age'])
        df_final=df_final.join(df['Driver_Total_Teenager_Age_15_19'])
        df_final=df_final.join(df['Driver_Total_College_Ages_20_23'])
        df_final=df_final.join(df['Driver_Total_Young_Adult_Ages_24_29'])
        df_final=df_final.join(df['Driver_Total_Low_Middle_Adult_Ages_30_39'])
        df_final=df_final.join(df['Driver_Total_Middle_Adult_Ages_40_49'])
        df_final=df_final.join(df['Driver_Total_Adult_Ages_50_64'])
        df_final=df_final.join(df['Driver_Total_Senior_Ages_65_69'])
        df_final=df_final.join(df['Driver_Total_Upper_Senior_Ages_70_plus'])
        
        
        
        df_final=df_final.join(df.Vehicle_Youthful_Driver_Indicator.str.get_dummies().add_prefix('Vehicle_Youthful_Driver_Indicator_'))
        df_final=df_final.join(df.Vehicle_Youthful_Driver_Training_Code.str.get_dummies().add_prefix('Vehicle_Youthful_Driver_Training_Code_'))
        df_final=df_final.join(df.Vehicle_Youthful_Good_Student_Code.str.get_dummies().add_prefix('Vehicle_Youthful_Good_Student_Code_'))
        
        df_final=df_final.join(df['Vehicle_Driver_Points'])
        
        df_final=df_final.join(df.Vehicle_Safe_Driver_Discount_Indicator.str.get_dummies().add_prefix('Vehicle_Safe_Driver_Discount_Indicator'))
        
        df_final=df_final.join(df['Annual_Premium'])
        #df_final=df_final.join(df['Loss_Amount'])
        
        df_final.to_csv('test_portfolio_cleaned_'+str(x)+'.csv')
    except:
        pass