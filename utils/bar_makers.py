import pandas as pd
import numpy as np
from typing import Union, Generator, Dict, Tuple
from numba import jit, njit, typed
# from numba.typed import Dict, List
# import numba as nb
from numba.core import types

@njit(parallel=True)
def extract_bars(data: np.ndarray, params: Dict, cum_values: Dict, threshold:float | np.ndarray, bar_type: str = 'dollar', is_threshold_array: bool = False) -> np.ndarray:
	bars_array = np.zeros((1, 10), dtype=np.float64)
	bar_type_mapping = {'dollar': 'cum_dollar_value', 'volume': 'cum_volume', 'tick': 'cum_ticks'}
	# thresh: float = threshold

	for i in range(0, len(data)):
		params['tick_num'] += 1
		date_time = data[i, 0]
		price = float(data[i, 1])
		volume = data[i, 2]
		dollar_value = price * volume
		tick_diff = 0

		if params['open_price'] == -1:
			params['open_price'] = price

		if params['prev_price'] == -1:
			tick_sign = 1
		else:
			tick_diff = price - params['prev_price']
			if tick_diff == 0:
				tick_sign = params['prev_tick_sign']
			else:
				tick_sign = np.sign(tick_diff)
		if not is_threshold_array:
			thresh = threshold

		if cum_values[bar_type_mapping[bar_type]] > thresh:
			new_bar = make_new_bar(params, cum_values)
			bars_array = np.vstack((bars_array, new_bar))
			reset_values(params, cum_values)

		params['high_price'], params['low_price'] = set_high_low(price, params)
		params['time_stamp'] = date_time
		params['close_price'] = price

		cum_values['cum_ticks'] += 1
		cum_values['cum_dollar_value'] += dollar_value
		cum_values['cum_volume'] += volume
		if tick_sign == 1:
			cum_values['cum_buy_volume'] += volume

		params['prev_price'] = price
		params['prev_tick_sign'] = tick_sign
	return bars_array

@njit(parallel=True)
def extract_time_bars(data: np.ndarray, params: Dict, cum_values: Dict, threshold: float):
	bars_ls = np.zeros((1,10), dtype=np.float64)
	for i in range(0, len(data)):
		params['tick_num'] += 1
		date_time = data[i, 0]
		price = float(data[i, 1])
		volume = data[i, 2]
		dollar_value = price * volume
		tick_diff = 0

		if params['open_price'] == -1:
			params['open_price'] = price
		if params['prev_price'] == -1:
			tick_sign = 1
		else:
			tick_diff = price - params['prev_price']
			if tick_diff == 0:
				tick_sign = params['prev_tick_sign']
			else:
				tick_sign = np.sign(tick_diff)
		
		timestamp_threshold = (date_time // threshold) * threshold
		if params['time_stamp'] == -1:
			params['time_stamp'] = timestamp_threshold
		
		if params['time_stamp'] < timestamp_threshold:
			new_bar = make_new_bar(params, cum_values)
			bars_ls = np.vstack((bars_ls, new_bar))
			reset_values(params, cum_values)
			params['time_stamp'] = timestamp_threshold


		params['high_price'], params['low_price'] = set_high_low(price, params)
		params['close_price'] = price
		cum_values['cum_ticks'] += 1
		cum_values['cum_dollar_value'] += dollar_value
		cum_values['cum_volume'] += volume
		
		params['prev_price'] = price
		params['prev_tick_sign'] = tick_sign

		if tick_sign == 1:
			cum_values['cum_buy_volume'] += volume

	return bars_ls

def run_in_batches(df: pd.DataFrame, threshold: np.float64 | np.ndarray, batch_size:np.float64, verbose: bool = True, bar_type: str = 'time') -> Union[pd.DataFrame, None]: 
	params = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	params['time_stamp'] = -1
	params['open_price'] = -1
	params['high_price'] = -np.inf
	params['low_price'] = np.inf
	# params['close_price'] = -1
	params['prev_price'] = -1
	params['prev_tick_sign'] = -1
	params['tick_num'] = 0
	
	cum_values = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	cum_values['cum_ticks'] = 0
	cum_values['cum_dollar_value'] = 0
	cum_values['cum_volume'] = 0
	cum_values['cum_buy_volume'] = 0

	if verbose:
		print('Reading data in batches:')

	count = 0
	final_bars = []
	cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume', 'cum_ticks',
			'cum_dollar_value']
	is_threshold_array: bool = isinstance(threshold, np.ndarray)
	for batch in batch_iterator(df, batch_size):
		if verbose:
			print('Batch number:', count)

		if bar_type == 'time':
			if not isinstance(threshold, float):
				raise ValueError("Time bars threshold can only be of type float not pd.Series")
			list_bars = extract_time_bars(batch.to_numpy(dtype=np.float64), params, cum_values, threshold)
		else:
			list_bars = extract_bars(batch.to_numpy(dtype=np.float64), params, cum_values, threshold, bar_type, is_threshold_array)
		final_bars += list(list_bars[1:])
		count += 1

	if verbose:	
		print('Returning bars \n')

	if final_bars:
		bars_df = pd.DataFrame(final_bars, columns=cols)
		bars_df = bars_df.reset_index(drop=True)
		return bars_df
	return None



def batch_iterator(df: pd.DataFrame, batch_size) -> Generator[pd.DataFrame, None, None]:
	batches = []
	for _, chunk in df.groupby(np.arange(len(df)) // batch_size):
		batches.append(chunk)

	for batch in batches:
		yield batch

@njit
def reset_values(params, cum_values):
	params['open_price'] = float(-1)
	# params['close_price'] = float(-1)
	params['high_price'], params['low_price'] = -np.inf, np.inf
	cum_values['cum_ticks'], cum_values['cum_dollar_value'] = float(0) , float(0)
	cum_values['cum_volume'], cum_values['cum_buy_volume'] = float(0) , float(0)
	cum_values['cum_volume'] = float(0)

    
@njit
def make_new_bar(params: Dict, cum_values: Dict) -> np.ndarray[(int,int), float]:
	open_price = params['open_price']
	high_price = params['high_price']
	low_price = params['low_price']
	close_price = params['close_price']

	tick_num = params['tick_num']
	date_time = params['time_stamp']
	
	high_price = max(high_price, open_price)
	low_price = min(low_price, open_price)
	volume = cum_values['cum_volume']
	cum_buy_volume = cum_values['cum_buy_volume']
	cum_ticks = cum_values['cum_ticks']
	cum_dollar_value = cum_values['cum_dollar_value']
	
	new_bar = [[date_time, tick_num, open_price,
				high_price, low_price, close_price,
				volume, cum_buy_volume, cum_ticks,
				cum_dollar_value]]
	return np.array(new_bar, dtype=np.float64)


@njit
def set_high_low(price: float, params:Dict) -> Tuple[float, float]:
	if price > params['high_price']:
		high_price = price
	else:
		high_price = params['high_price']

	if price < params['low_price']:
		low_price = price
	else:
		low_price = params['low_price']
	return high_price, low_price



def get_bars(df: pd.DataFrame, resolution: str='M', num_units: int=1,
				threshold: float | pd.Series | np.ndarray | None = None, batch_size: float = 1e6,
				verbose: bool = True, bar_type: str = 'time') -> pd.DataFrame:
	df = df.dropna()
	batch_size = float(batch_size)
	df = df[['date_time', 'price', 'volume']]
	thresh = float(0)
	if isinstance(threshold, pd.Series):
		assert len(threshold) == len(df), 'Threshold must be the same length as the dataframe'
		thresh = np.array(threshold.values, dtype=np.float64)
	elif isinstance(threshold, np.ndarray):
		thresh = threshold
	else:
		if bar_type == 'time' and threshold is None:
			time_bar_thresh_mapping = {'D': 86400, 'H': 3600, 'M': 60, 'S': 1}  # Number of seconds
			thresh = float(num_units * time_bar_thresh_mapping[resolution] * 1e9)
		elif bar_type == 'time' and threshold is not None:
			raise ValueError('Threshold is not allowed for bar_type == time use num_units and resolution instead')
		elif bar_type != 'time' and threshold is None:
			raise ValueError('Threshold must be provided for bar_type != time')
		else:
			thresh = float(threshold)
	
	bars = run_in_batches(df, thresh, batch_size, verbose, bar_type)
	if bars is None:
		raise ValueError("Extracted bars is None")
	# if bar_type == 'time':
	# 	bars['date_time'] = pd.to_datetime(bars['date_time'], unit='ns')
	# else:
	times = pd.to_datetime(bars['date_time'], unit='ns').dt.strftime("%Y-%m-%d %H:%M:%S")
	bars['date_time'] = pd.to_datetime(times)
	bars['cum_ticks'] = bars['cum_ticks'].astype(np.int64)
	bars['tick_num'] = bars['tick_num'].astype(np.int64)
	return bars
