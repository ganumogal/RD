from sklearn.datasets import load_iris
import pandas as pd
data=pd.read_csv("iris.csv")
print(data)
#isnull() function
print(data.isnull())
#is there any missing value along each column
print(data.isnull().any())
# count of missing values across each column using isna() and isnull()
print(data.isnull().sum().sum())
#count row wise missing value using isnull()
print(data.isnull().sum(axis = 1))
#count Column wise missing value using isnull()
print(data.isnull().sum())
#b
print(data.isna().sum())
#count of missing values of a specific column
print(data.SepalLengthCm.isnull().sum())
#groupby count of missing values of a column.
print(data.groupby(['SepalLengthCm'])['PetalWidthCm'].apply(lambda x:
x.isnull().sum()))

#Panda functions for Data Formatting and Normalization
iris = load_iris()
#create a pandas DataFrame with the iris data
data = pd.DataFrame(iris.data,
columns=iris.feature_names)
#load iris dataset
print(data.head())
#unique values for species column
print(data['sepal width (cm)'].unique())
# one_hot_data_encoding
one_hot_data = pd.get_dummies(data, prefix="sepal width (cm)",
columns=['sepal width (cm)'], drop_first=False)
print(one_hot_data)