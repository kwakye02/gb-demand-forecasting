# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import  Path
import glob
from statsmodels.graphics.tsaplots import plot_acf
import holidays

pd.set_option("display.max_columns", None)

plt.style.use('dark_background')
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['grid.alpha'] = 0.8

PROJECT_ROOT = '/Users/kwakye/Desktop/Github Projects/portfolio_project_one/gb-demand-imbalance-forecasting/data/processed/*.parquet'

files = glob.glob(PROJECT_ROOT)

for f in files:
    print(f)

#load the demand and weather datasets
df_demand = pd.read_parquet(files[0])
df_weather = pd.read_parquet(files[1])


#check shape and info of the dataset
df_demand.shape
df_weather.shape

#check info of dataset
df_demand.info()
df_weather.info()

df_demand = df_demand[['SETTLEMENT_DATE','SETTLEMENT_PERIOD','ND','EMBEDDED_WIND_GENERATION','EMBEDDED_SOLAR_GENERATION','ENGLAND_WALES_DEMAND']].copy()

#check first few rows of dataset
df_demand.head(5)

#convert settlement_date to a datetime object
df_demand["SETTLEMENT_DATE"] = pd.to_datetime(df_demand["SETTLEMENT_DATE"],format="mixed", dayfirst=True,utc=True)

# Create a full timestamp
df_demand['datetime'] = df_demand['SETTLEMENT_DATE'] + pd.to_timedelta((df_demand['SETTLEMENT_PERIOD'] -1)*30,unit ='minutes')
df_demand[["SETTLEMENT_DATE",'SETTLEMENT_PERIOD','datetime']].head()
df_demand = df_demand.sort_values("datetime").reset_index(drop=True)

#merge the demand and weather datasets on the datetime column
df_weather['datetime'] = pd.to_datetime(df_weather['time'])
df_weather = df_weather.sort_values("datetime").reset_index(drop=True)
df = pd.merge(df_demand,df_weather,on='datetime',how='left')

#check missing values in the merged dataset
df.isnull().sum()

#create time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_of_year"] = df["datetime"].dt.dayofyear
df['quarter'] = df['datetime'].dt.quarter
df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)

#add holidays columms
uk_holidays = holidays.UnitedKingdom(years = range(2009,2026))
holiday_dates = pd.to_datetime(list(uk_holidays.keys()))
df["is_holiday"] = df["datetime"].dt.normalize().isin(holiday_dates).astype(int)

#set datetime as index
df = df.set_index('datetime')

df.tail(10)


#check missing values in the dataset
df[df.isna().any(axis=1)]

#ffill missing values in the dataset
df = df.ffill()

#check missing values in the dataset after ffill
df.isnull().sum()

#check summary statistics of the dataset
df.describe().round(2)

#check summary statistics for the demand column for each year
df[['ND','year']].groupby(by ='year')['ND'].agg(Mean ='mean',STD ='std', Max ='max',Min ='min').round()


#check for duplicates
duplicate_rows  = df[df.duplicated()]
print(duplicate_rows)


#check demand distribution 
df['ND'].hist()
plt.title("Distribution of National Demand (MW)")
plt.ylabel("Frequency")
plt.show()


ax = df[['ND','ENGLAND_WALES_DEMAND']].plot(subplots=True)
ax[0].set_title("National Demand (MW)")
ax[0].set_ylabel("ND (MW)")
ax[1].set_title("England & Wales Demand (MW)")
ax[1].set_ylabel("EW Demand (MW)")
plt.show()

#daily pattern 
hourly_demand = df.groupby('hour')['ND'].mean()

hourly_demand.plot()
plt.title("Average National Demand by Hour of Day")
plt.ylabel("Average National Demand (MW)")
plt.show()

#Demand was quite low during the periods between 3 am to 5 am
#Demand started rising after 5 am and plateaued at 10 am before rising and peaking at around 6pm
# demand kept falling from evening to morning 


plt.figure(figsize = (10,6))
sns.boxplot(data = df,x = 'hour', y='ND',)
plt.title('Demand by hour of the day')
plt.show()


#weekly pattern
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
daily_avg = df.groupby('day_of_week')['ND'].mean()

#plotting the average demand for each day of the week
days = ["Mon", "Tue", "Wed","Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(8, 5))
daily_avg.plot(kind="bar", ax=ax)
ax.set_xticks(range(len(days)))
ax.set_xticklabels(days, rotation=0)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average National Demand (MW)")
ax.set_title("Average GB Electricity Demand by Day of Week")

plt.tight_layout()
plt.show()

#check hourly average demand for weekdays and weekends
weekday_hourly = df[df["is_weekend"] == 0].groupby("hour")["ND"].mean()
weekend_hourly = df[df["is_weekend"] == 1].groupby("hour")["ND"].mean()
fig, ax = plt.subplots(figsize = (10,6))
weekday_hourly.plot(ax=ax, label= 'Weekday')
weekend_hourly.plot(ax=ax,label= 'weekend')
ax.set_title("Average National Demand by Hour: Weekday vs Weekend")
ax.set_ylabel("Average National Demand (MW)")
plt.legend()
plt.show()


#monthly pattern
monthly_avg = df.groupby("month")["ND"].mean()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize =(10,5))
monthly_avg.plot(ax=ax)
ax.set_xticks(monthly_avg.index)
ax.set_xticklabels(months, rotation = 0)
ax.set_title("Average National Demand by Month")
ax.set_ylabel("Average National Demand (MW)")
plt.show()

#check quarterly
quarterly_avg = df.groupby('quarter')['ND'].mean()
fig, ax = plt.subplots(figsize=(8, 5))
quarterly_avg.plot(kind = 'bar', ax=ax)
ax.set_xticks(range(4))
ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], rotation=0)
ax.set_title("Average National Demand by Quarter")
ax.set_ylabel("Average National Demand (MW)")
plt.show()


#yearly demand 
yearly_avg = df.groupby("year")["ND"].mean()


fig, ax = plt.subplots(figsize=(12,5))

yearly_avg.plot(ax=ax,marker="o")

ax.set_title("Average GB Electricity Demand by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Average National Demand (MW)")
plt.show()


yearly_avg_wind = df.groupby("year")["EMBEDDED_WIND_GENERATION"].mean()
yearly_avg_solar = df.groupby('year')['EMBEDDED_SOLAR_GENERATION'].mean()


fig, ax = plt.subplots(figsize=(12,5),nrows=3, ncols= 1, sharex=True)
yearly_avg.plot(ax=ax[0],marker="o", label = 'ND')
ax[0].set_title('Average National Demand')
ax[0].set_ylabel('National Demand')
yearly_avg_solar.plot(ax=ax[1],marker="o",label ='Solar')
ax[1].set_ylabel('Solar')
ax[1].set_title('Average Embedded Solar Generation')
yearly_avg_wind.plot(ax=ax[2],marker="o",label ='Wind')
ax[2].set_ylabel(' Wind')
ax[2].set_title('Average Embedded Wind Generation')
ax[2].set_xlabel("Year")
plt.show()

#Plot demand vs weather variables
def plot_demand_vs_weather(df, variable_list):
    length = len(variable_list)
    fig, axes = plt.subplots(figsize=(10,6*length), nrows=length, ncols=1)
    for i,variable in enumerate(variable_list):
        sns.scatterplot(data=df, x=variable, y="ND",ax=axes[i])
        axes[i].set_title(variable + " vs National Demand")
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel("National Demand (MW)")
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel("National Demand (MW)")

    plt.tight_layout()
    plt.show()

variable_list =["relative_humidity_2m","temperature_2m","precipitation",
                "cloud_cover","wind_speed_10m","shortwave_radiation",]
plot_demand_vs_weather(df, variable_list)



# check correlation
col = ['ND', 'EMBEDDED_WIND_GENERATION', 'EMBEDDED_SOLAR_GENERATION',
     'temperature_2m', 'relative_humidity_2m', 'precipitation', 'cloud_cover',
     'wind_speed_10m', 'shortwave_radiation', 'year', 'month']

plt.figure(figsize=(15,15))
sns.heatmap(df[col].corr(),annot=True,fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#add lag features
plt.figure(figsize=(30, 20))
plot_acf(df["ND"],lags= 100)
plt.title("Autocorrelation Function of National Demand")
plt.xlabel("Lag (30-minute intervals)")
plt.ylabel("Autocorrelation")
plt.show()

#create lags of 24h and weekly
df['lag_24h'] = df['ND'].shift(48)
df['lag_168h'] = df['ND'].shift(336)

df.iloc[329:350].head(30)

#rolling features 
df['roll_mean_24h'] = df['ND'].shift(1).rolling(window = 48).mean()
df['roll_mean_168h'] = df['ND'].shift(1).rolling(window = 336).mean()
df['roll_std_24h'] = df['ND'].shift(1).rolling(window = 48).std()
df['roll_std_168h'] = df['ND'].shift(1).rolling(window = 336).std()

df.isnull().sum()
#drop na values 
data = df.dropna().copy()

#drop features not being used
data.drop(columns=['SETTLEMENT_DATE','EMBEDDED_WIND_GENERATION','EMBEDDED_SOLAR_GENERATION','ENGLAND_WALES_DEMAND'],inplace=True)


data.isnull().sum()

