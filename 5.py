
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
df1=(pd.read_csv('social.csv'))
print(df1)
print(df1.head())
print(df1.describe())
plt.scatter(df1['Age'],df1['Purchased'])

plt.xlabel("Age")
plt.ylabel("Purchased")
plt.show()
"/////////Dividing the data into training set and test set/////////////"
X_train, X_test, Y_train, Y_test = train_test_split(df1['Age'], df1['Purchased'],test_size=0.20)
"////////////////making prediction using scikit learn//////////////// "
lr_model=LogisticRegression()
lr_model.fit(X_train.values.reshape(-1,1),(Y_train.values.reshape(-1,1).ravel()))
Y_pred_sk=lr_model.predict(X_test.values.reshape(-1,1))
plt.clf()
plt.scatter(X_test,Y_test)
plt.scatter(X_test,Y_pred_sk,c='red')        
plt.xlabel('Age')   
plt.ylabel('purchased')
plt.show() 
print(f"Accuracy={lr_model.score(X_test.values.reshape(-1,1),(Y_test.values.reshape(-1,1)))}")
print("////////////////////confusion matrix///////////////////////")
tn,fn,tp,fp=confusion_matrix(Y_test,Y_pred_sk).ravel()
print("true negative",tn)
print("true positive",tp)
print("false positive",fp)
print("false negative",fn)
Accuracy=(tn+tp)*100/(tn+tp+fn+fp)    
print("Accuracy {:0.2f}%". format(Accuracy))       
Precision=tp/fp+tp
print("Precision{:0.2f}%". format(Precision))
Recall=tp/tp+fn 
print("Recall{:0.2f}%". format(Recall))
err=(fp+fn)/(fp+fn+tp+tn)
print("err{:0.2f}%". format(err))
                                                                                                         