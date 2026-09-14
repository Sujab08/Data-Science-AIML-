import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\kumar\Documents\CollegePlacement.csv")
print(df.info())
print(df.head(10))
print(df.describe())
co=df[["IQ","Prev_Sem_Result","CGPA","Academic_Performance","Extra_Curricular_Score","Communication_Skills","Projects_Completed"]].corr()
plt.imshow(co,cmap="coolwarm",interpolation="none")
plt.colorbar()
plt.xticks(range(len(co)),co.columns,rotation=45)
plt.yticks(range(len(co)),co.columns)
plt.show()