import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")
df.info()
print(df.describe())
print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df['Species'].unique())

df_setosa = df.loc[df['Species']=='Iris-setosa']
df_versicolor = df.loc[df['Species']=='Iris-versicolor']
df_virginica = df.loc[df['Species']=='Iris-virginica']
print(df_setosa.head())


plt.plot(df_setosa['SepalLengthCm'],np.zeros_like(df_setosa['SepalLengthCm']),'o')
plt.plot(df_versicolor['SepalLengthCm'],np.zeros_like(df_versicolor['SepalLengthCm']),'o')
plt.plot(df_virginica['SepalLengthCm'],np.zeros_like(df_virginica['SepalLengthCm']),'o')
plt.xlabel("SepalLengthCm")
plt.show()

sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=df,hue='Species')
plt.show()

sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=df,hue='Species')
plt.show()

df_no_id_clmn=df.drop('Id',axis=1)
print(df_no_id_clmn.head())