
import holidays

def create_features(dataframe):
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df['quarter'] = df['datetime'].dt.quarter
    df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)
    #add holidys columms
    uk_holidays = holidays.UnitedKingdom(years = range(2009,2026))
    holiday_dates = pd.to_datetime(list(uk_holidays.keys()))
    df["is_holiday"] = df["datetime"].dt.normalize().isin(holiday_dates).astype(int)

    #rolling features 
    df['roll_mean_24h'] = df['ND'].shift(1).rolling(window = 48).mean()
    df['roll_mean_168h'] = df['ND'].shift(1).rolling(window = 336).mean()
    df['roll_std_24h'] = df['ND'].shift(1).rolling(window = 48).std()
    df['roll_std_168h'] = df['ND'].shift(1).rolling(window = 336).std()
    return df