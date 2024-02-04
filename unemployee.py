import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
import calendar
import plotly.express as px

df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
print(df.head())
print(df.tail())
df.info()
print(df.describe())

print(df.columns)
df.columns = ['State', 'Date', 'Frequency', 'Estimated Unemployment Rate (%)',
              'Estimated Employed', 'Estimated Labour Participation Rate (%)',
              'Region', 'longitude', 'latitude']
print(df.head())

print(df.isnull().sum())
print(df.duplicated().sum())
print(df.State.value_counts())
df['Date']=pd.to_datetime(df['Date'],dayfirst=True)
print(df.info())
df['month_num'] = df['Date'].dt.month
print(df.head())
df['month'] = df['month_num'].apply(lambda x: calendar.month_abbr[x])
print(df.head())

sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Unemployment Rate (%)'], bins=30, kde=True)
plt.title('Distribution of Estimated Unemployment Rate (%)')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Employed'], bins=30, kde=True)
plt.title('Distribution of Estimated Employed')
plt.xlabel('Number of Employed')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Labour Participation Rate (%)'], bins=30, kde=True)
plt.title('Distribution of Estimated Labour Participation Rate (%)')
plt.xlabel('Labour Participation Rate (%)')
plt.ylabel('Frequency')
plt.show()

fig = px.histogram(df, x='State', y='Estimated Unemployment Rate (%)',color='State', animation_frame='month',
                   labels={'Estimated Unemployment Rate (%)': 'Unemployment Rate (%)'},
                   title='Monthly Unemployment Rate Across States')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.show()

df_sorted = df.sort_values(by='Date')

# Plotting the unemployment rate over time
plt.figure(figsize=(12, 6))
plt.plot(df_sorted['Date'], df_sorted['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='b')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.grid(True)
plt.show()

State =  df.groupby(['State'])[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].mean()
State = pd.DataFrame(State).reset_index()

fig = px.box(data_frame=df,x='State',y='Estimated Unemployment Rate (%)',color='State',title='Unemployment rate')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Estimated Unemployment Rate (%)', y='Estimated Labour Participation Rate (%)')
plt.title('Unemployment Rate vs Labour Participation Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Labour Participation Rate (%)')
plt.show()

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', label='Unemployment Rate')
sns.lineplot(data=df, x='Date', y='Estimated Labour Participation Rate (%)', label='Labour Participation Rate')
plt.title('Unemployment Rate and Labour Participation Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.legend()
plt.show()

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', label='Unemployment Rate')
sns.lineplot(data=df, x='Date', y='Estimated Labour Participation Rate (%)', label='Labour Participation Rate')
plt.title('Unemployment Rate and Labour Participation Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.legend()
plt.show()

sns.pairplot(df, vars=['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)'])
plt.suptitle('Pairplot of Unemployment Rate, Employed Count, and Labour Participation Rate', y=1.02)
plt.show()

plt.figure(figsize=(10, 8))
corr = df[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

unemployment =df.groupby(['Region','State'])['Estimated Unemployment Rate (%)'].mean().reset_index()
unemployment.head()
fig = px.sunburst(unemployment,path=['Region','State'],values='Estimated Unemployment Rate (%)',
                 title ='Unemployment rate in state and region',height=600)
fig.show()