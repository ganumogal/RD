
import matplotlib.pyplot as pyplot
import seaborn as sns
dataset=sns.load_dataset('titanic')
print(dataset.head())
sns.displot(dataset['fare'],kde=False,bins=10)
sns.jointplot(x='age',y='fare',data=dataset,kind='hex')
sns.rugplot(dataset['fare'])
pyplot.show()