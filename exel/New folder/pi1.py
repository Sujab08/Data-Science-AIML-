from sklearn.tree import DecisionTreeClassifier
X=[[7,2],[8,3],[9,8],[10,9]]
y=[0,0,1,1]
model=DecisionTreeClassifier()
model.fit(X,y)
size=float(input("enter size:"))
color=float(input("enter color:"))
result=model.predict([[size,color]])[0]
if result==1:
    print("orange")
else:
    print("mango")