import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv('StudentsPerformance.csv')
print(df)
print(df.isnull())
series=pd.isnull(df['Math_Score'])
print(df[series])
print(df.notnull())
ndf=df
col_drop=ndf.dropna(axis=1)
print(col_drop)
row_drop=ndf.dropna(axis=0,how='any')
print(row_drop)
col=['Math_Score','Reading_Score','Writing_Score','Placement_Score']
df.boxplot(col)
print(np.where(df['Math_Score']>90))
print(np.where(df['Reading_Score']<25))
print(np.where(df['Writing_Score']<30))
fig,ax=plt.subplots(figsize=(18,10))
ax.scatter(df['Placement_Score'],df['Placement_Offer_Count'])
plt.show()
print(np.where((df['Placement_Score']<50)))
print(np.where((df['Placement_Score']>85)))
z=np.abs(stats.zscore(df['Math_Score']))
print(z)
threshold=0.18
sample_outliers=np.where(z<threshold)
print(sample_outliers)