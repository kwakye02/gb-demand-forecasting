import holidays
import pandas as pd

FEATURE_COLS = [
    "lag_2d", "lag_7d", "lag_14d", "lag_21d", "lag_28d", "lag_35d", "lag_42d",
    "prev_6weeks_lag_noholiday", "lag7d_minus_prev_6weeks",
    "d_solar_7", "d_wind_7", "d_solar_base", "d_wind_base",
]


def add_holiday_flags(df):
    """xmas_period, day_before_holiday and day_after_holiday, from the England bank-holiday calendar."""
    df = df.copy()
    calendar = holidays.UnitedKingdom(subdiv="ENG", years=range(2019, pd.Timestamp.now().year + 2))
    hol = pd.to_datetime(sorted(calendar.keys()))
    d = df["local_date"]
    df["xmas_period"] = (((d.dt.month == 12) & (d.dt.day >= 24)) | ((d.dt.month == 1) & (d.dt.day == 1))).astype(int)
    df["day_before_holiday"] = (d + pd.Timedelta(days=1)).isin(hol).astype(int)
    df["day_after_holiday"] = (d - pd.Timedelta(days=1)).isin(hol).astype(int)
    return df


def daily_table(df, col):
    """col by (local_date, local_time), with gaps of up to two half-hours filled (clock-change days)."""
    table = df.groupby(["local_date", "local_time"])[col].mean().unstack("local_time")
    table = table.reindex(columns=sorted(table.columns))
    return table.interpolate(axis=1, limit=2, limit_area="inside").stack()


def lag_by_days(df, table, days):
    """Value of the table at the same local time, `days` days earlier."""
    keys = pd.MultiIndex.from_arrays(
        [df["local_date"].dt.normalize() - pd.Timedelta(days=days), df["local_time"]]
    )
    return table.reindex(keys).to_numpy()


def weekly_lags(df, table, weeks=6):
    """Same weekday and time 1..weeks weeks earlier, and their mean with bank-holiday dates left out."""
    day = df["local_date"].dt.normalize()
    hol_days = set(day[df["is_holiday"] == 1])
    lags, kept = {}, []
    for k in range(1, weeks + 1):
        lag = pd.Series(lag_by_days(df, table, 7 * k), index=df.index)
        lags[f"lag_{7 * k}d"] = lag
        kept.append(lag.where(~(day - pd.Timedelta(days=7 * k)).isin(hol_days)))
    return pd.DataFrame(lags), pd.concat(kept, axis=1).mean(axis=1)


def build_features(df):
    """Add every notebook-03 feature to a merged frame (needs ND, emb_*_fc, is_holiday, local_date, local_time)."""
    df = add_holiday_flags(df)

    demand = daily_table(df, "ND")
    df["lag_2d"] = lag_by_days(df, demand, 2)
    lags, baseline = weekly_lags(df, demand)
    df[lags.columns] = lags
    df["prev_6weeks_lag_noholiday"] = baseline
    df["lag7d_minus_prev_6weeks"] = df["lag_7d"] - baseline

    for name, col in [("solar", "emb_solar_fc"), ("wind", "emb_wind_fc")]:
        table = daily_table(df, col)
        df[f"d_{name}_7"] = df[col] - lag_by_days(df, table, 7)
        df[f"d_{name}_base"] = df[col] - weekly_lags(df, table)[1]
    return df