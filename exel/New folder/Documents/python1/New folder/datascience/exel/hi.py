import pandas as pd
df=pd.read_excel(r"C:\Users\kumar\Documents\Copy of Format_for_Mapping_254_Mandal_Sir(1).xlsx")
print(df["Unnamed: 6"].value_counts())    
print(df.info())             
