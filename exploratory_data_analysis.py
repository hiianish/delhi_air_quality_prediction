import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("final_dataset")

print("shape",df.shape)
df.info()
df.describe()
df.isnull().sum()

#Correlation Analysis (Which pollutant affects AQI most?)
plt.figure(figsize=(10,6))
sns.heatmap(df[['PM2.5','PM10','NO2','SO2','CO','Ozone','AQI']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap between Pollutants and AQI")
plt.show()



#Temporal Trend of AQI
plt.figure(figsize=(12,5))
plt.plot(df['Date_time'], df['AQI'], color='red')
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Air Quality Index")
plt.show()



#Distribution of Each Pollutant
pollutants = ['PM2.5','PM10','NO2','SO2','CO','Ozone']

plt.figure(figsize=(15,8))
for i, col in enumerate(pollutants, 1):
    plt.subplot(2,3,i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


#Seasonal Variation
plt.figure(figsize=(8,5))
sns.boxplot(x='Season', y='AQI', data=df, order=['Winter','Summer','Monsoon','Post-Monsoon'])
plt.title("Seasonal Variation of AQI")
plt.show()



#Monthly & Yearly Trends
#Month
monthly_aqi = df.groupby('Month')['AQI'].mean()
plt.figure(figsize=(10,5))
monthly_aqi.plot(kind='bar', color='darkorange')
plt.title("Average AQI by Month")
plt.ylabel("AQI")
plt.show()

#Year
yearly_aqi = df.groupby('Year')['AQI'].mean()
plt.figure(figsize=(8,4))
yearly_aqi.plot(kind='bar', color='teal')
plt.title("Average AQI by Year")
plt.ylabel("AQI")
plt.show()



#Weekday vs Weekend Patterns
plt.figure(figsize=(8,5))
sns.boxplot(x='is_weekend', y='AQI', data=df)
plt.title("AQI: Weekend (1) vs Weekday (0)")
plt.xlabel("Is Weekend?")
plt.ylabel("AQI")
plt.show()




#Effect of Holidays
plt.figure(figsize=(8,5))
sns.boxplot(x='is_holiday', y='AQI', data=df)
plt.title("AQI on Holidays vs Normal Days")
plt.xlabel("Is Holiday?")
plt.ylabel("AQI")
plt.show()



#Pollutant Impact on AQI
for col in pollutants:
    plt.figure(figsize=(6,4))
    sns.regplot(x=col, y='AQI', data=df, scatter_kws={'alpha':0.3})
    plt.title(f"AQI vs {col}")
    plt.show()




#PCA (Principal Component Analysis)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = ['PM2.5','PM10','NO2','SO2','CO','Ozone']
x_scaled = StandardScaler().fit_transform(df[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_scaled)

plt.figure(figsize=(7,5))
plt.scatter(pca_result[:,0], pca_result[:,1], c=df['AQI'], cmap='coolwarm', alpha=0.6)
plt.colorbar(label='AQI')
plt.title("PCA: Pollutant Pattern Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
