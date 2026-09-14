import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data={
    'name':['sujab','mandal','kumar','kapli','om'],
    'age':[25,None,44,23,None],
    'salary':[50000,60000,70000,None,None]
}
from sklearn.linear_model import LogisticRegression
X=[[1],[2],[3],[4],[5]]
y=[0,0,1,1,1]
model=LogisticRegression()
model.fit(X,y)
hour=float(input("enter how many hours you have studied:"))
result=model.predict([[hour]])[0]
if result==1:
    print("pass")
else:
    print("fail")      
from sklearn.neighbors import KNeighborsClassifier
X=[[180,7],[200,7.5],[250,8],[300,8.5],[330,9],[360,9.5]]
y=[0,0,0,1,1,1,1]
model=KNeighborsClassifier()
model.fit(X,y)
weight=float(input("enter weight:"))
size=float(input("enter size:"))
result=model.predict([[weight,size]])[0]
if result==1:
    print("orange")
else:
    print("mengo")
