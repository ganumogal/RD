import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('titanic.csv')
print(df)
cols=df.columns
print(cols)
print(df.info())
print(df.describe())
print(df.isnull().sum())
sns.boxplot(x='sex', y='age', data=df);
sns.boxplot(x='sex', y='age', data=df, hue="survived");
plt.show()