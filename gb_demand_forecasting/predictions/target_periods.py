from datetime import date, timedelta
import pandas as pd


def get_target_periods(target_date):
    """ Generate all settlement periods for a given target date."""

    day = pd.Timestamp(target_date)

    start = day.tz_localize('Europe/London')
    end = (day + pd.Timedelta(days=1)).tz_localize('Europe/London')

    timestamps = pd.date_range(start=start.tz_convert("UTC"), end=end.tz_convert("UTC"),
                                freq='30min', inclusive='left')

    local_times = timestamps.tz_convert('Europe/London')

    # Create a DataFrame with the local times and their corresponding settlement periods
    df = pd.DataFrame({
        "SETTLEMENT_DATE":day,
        "SETTLEMENT_PERIOD": range(1, len(local_times) + 1),
        "datetime_local": local_times,
        "local_date": day,
        "local_time":local_times.strftime('%H:%M')
    },
    index = timestamps.rename('datetime')
    )

    return df