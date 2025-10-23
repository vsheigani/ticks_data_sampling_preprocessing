import pandas as pd
import numpy as np

def create_metric_threshold_series(df, metric="dollar", window=2, resolution='M', bars_per_day=50):
    assert resolution.upper() in ['ME', 'YE'], 'Resolution must be one of the following: ME (Month End), YE (Year End)'
    assert window > 0, 'Window must be greater than 0'
    assert metric in ['dollar', 'volume', 'tick'], 'Metric must be one of the following: dollar, volume, tick'

    business_days_mapping = {'ME': 21, 'YE': 252}
    message_mapping = {'ME': 'Months', 'YE': 'Years'}
    print(f'Creating threshold series for past {window} {message_mapping[resolution]} windows')
    tmp_df = pd.DataFrame(index=df.index)
    tmp_df['idx'] = df.index
    tmp_df['date_time'] = df['date_time']

    if metric == 'dollar':
        tmp_df['value'] = (df.price * df.volume)
    elif metric == 'volume':
        tmp_df['value'] = df.volume
    elif metric == 'tick':
        tmp_df['value'] = 1

    tmp_df.set_index('date_time', inplace=True)
    # add the values of the requested metric for all the ticks in each month
    resampled_df = tmp_df.resample(resolution).agg({'value': "sum", 'idx':'last'})
    # number of bars per day = average number of business days per resolution * expected number of bars per day
    expected_num_bars = business_days_mapping[resolution] * bars_per_day
    resampled_df['value'] = np.round(resampled_df['value'].rolling(window).mean() / expected_num_bars, 2)
    resampled_df = resampled_df.set_index('idx')

    threshold = pd.Series(data=resampled_df['value'], index=df.index)
    threshold = threshold.ffill().bfill().to_numpy(dtype=np.float64).flatten()
    return pd.Series(threshold).shift(window-1).bfill()

def calculate_volume_threshold_constant(df, lookback=None, num_bars_per_day=50):
    # The number of rows at the end of the dataframe that you want to use for calculating threshold
    period = lookback if lookback else 10000
    if period > df.shape[0]:
        period = df.shape[0]
    start_idx = int(df.shape[0]//2 - period//2)
    end_idx = int(df.shape[0]//2 + period//2)
    temp_series = df.iloc[start_idx:end_idx, :].copy(deep=True)
    temp_series = temp_series.set_index('date_time').resample(rule='D').sum()
    threshold = np.round(temp_series.volume.mean()/float(num_bars_per_day), 2)
    print(f'volume threshold: {threshold}')
    return threshold

def calculate_tick_threshold_constant(df, lookback=None, num_bars_per_day=50):
    # The number of rows at the end of the dataframe that you want to use for calculating threshold
    period = lookback if lookback else 10000
    if period > df.shape[0]:
        period = df.shape[0]
    start_idx = int(df.shape[0]//2 - period//2)
    end_idx = int(df.shape[0]//2 + period//2)
    temp_series = df.iloc[start_idx:end_idx, :].copy(deep=True)
    temp_series = temp_series.set_index('date_time').resample(rule='D').count().price
    threshold = np.round(temp_series.mean()/float(num_bars_per_day), 2)
    print(f'tick threshold: {threshold}')
    return threshold

def calculate_dollar_threshold_constant(df, lookback=None, num_bars_per_day=50):
    # The number of rows at the end of the dataframe that you want to use for calculating threshold
    period = lookback if lookback else 10000
    if period > df.shape[0]:
        period = df.shape[0]
    start_idx = int(df.shape[0]//2 - period//2)
    end_idx = int(df.shape[0]//2 + period//2)
    temp_series = df.iloc[start_idx:end_idx, :].copy(deep=True)
    temp_series = temp_series.set_index('date_time')
    
    calc_series = pd.Series(index=temp_series.index, dtype='float64')
    calc_series = temp_series['price'] * temp_series['volume']

    threshold = np.round(calc_series.resample(rule='D').sum().mean()/num_bars_per_day, 2)
    print(f'Dollar Threshold: {threshold}')
    return threshold