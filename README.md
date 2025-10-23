# Ticks Data Sampling & Preprocessing

A high-performance Python package for converting high-frequency tick data into various bar types (time, volume, dollar, and tick bars) for financial analysis and algorithmic trading.

## Overview

This package provides efficient tools for preprocessing high-frequency financial tick data into different bar representations. It's designed to handle large datasets with optimized batch processing using Numba JIT compilation for maximum performance.

## Features

- **Multiple Bar Types**: Support for time bars, volume bars, dollar bars, and tick bars
- **High Performance**: Optimized with Numba JIT compilation for fast processing
- **Batch Processing**: Memory-efficient processing of large datasets
- **Statistical Analysis**: Built-in normality tests and return distribution analysis
- **Visualization**: Integrated plotting capabilities for bar analysis
- **Flexible Thresholds**: Support for both constant and variable thresholds

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ticks_data_sampling_preprocessing

# Install with uv
uv sync
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from utils.bar_makers import get_bars
from utils.threshold_calculators import calculate_dollar_threshold_constant

# Load tick data
ticks_df = pd.read_csv("ticks_data_ns/googl_trade_2023_05_12.txt", 
                       sep=",", header=None, 
                       names=["date_time", "price", "volume", "exchange_code", "trading_condition"])

# Convert to datetime
ticks_df['date_time'] = pd.to_datetime(ticks_df['date_time'], unit='ns')
ticks_df = ticks_df.drop(columns=["exchange_code", "trading_condition"])

# Create dollar bars
dollar_threshold = calculate_dollar_threshold_constant(ticks_df, lookback=1e6, num_bars_per_day=50)
dollar_bars = get_bars(ticks_df, threshold=dollar_threshold, batch_size=1e6, bar_type='dollar')
```

### Creating Different Bar Types

#### Time Bars
```python
# 30-minute time bars
time_bars = get_bars(ticks_df, resolution='M', num_units=30, bar_type='time')
```

#### Volume Bars
```python
volume_threshold = calculate_volume_threshold_constant(ticks_df, lookback=1e6, num_bars_per_day=50)
volume_bars = get_bars(ticks_df, threshold=volume_threshold, bar_type='volume')
```

#### Tick Bars
```python
tick_threshold = calculate_tick_threshold_constant(ticks_df, lookback=1e6, num_bars_per_day=50)
tick_bars = get_bars(ticks_df, threshold=tick_threshold, bar_type='tick')
```

## Data Format

The library expects tick data in the following format:

| Column | Description | Type |
|--------|-------------|------|
| date_time | Timestamp in nanoseconds | datetime |
| price | Trade price | float |
| volume | Trade volume | float |

## Bar Types Explained

### Time Bars
- **Purpose**: Aggregate data over fixed time intervals
- **Use Case**: Traditional technical analysis, regular market hours analysis
- **Parameters**: `resolution` ('D', 'H', 'M', 'S') and `num_units`

### Volume Bars
- **Purpose**: Create bars when a certain volume threshold is reached
- **Use Case**: Volume-based analysis, liquidity studies
- **Parameters**: Volume threshold value

### Dollar Bars
- **Purpose**: Create bars when a certain dollar value threshold is reached
- **Use Case**: Dollar-weighted analysis, institutional trading patterns
- **Parameters**: Dollar threshold value

### Tick Bars
- **Purpose**: Create bars when a certain number of ticks is reached
- **Use Case**: Tick-based analysis, microstructure studies
- **Parameters**: Tick count threshold

## Performance Features

- **Numba JIT Compilation**: Core functions are compiled for maximum speed
- **Batch Processing**: Handles large datasets by processing in chunks
- **Memory Efficient**: Optimized memory usage for large tick datasets
- **Parallel Processing**: Utilizes multiple CPU cores where possible

## Statistical Analysis

The library includes built-in statistical analysis tools:

```python
from scipy import stats
import numpy as np

# Calculate returns
returns = np.log(1 + bars['close'].pct_change()).dropna()

# Jarque-Bera normality test
result = stats.jarque_bera(returns)
print(f'Statistics: {result.statistic}, p-value: {result.pvalue}')
```

## Visualization

```python
import matplotlib.pyplot as plt
import mplfinance as mpf

# Plot candlestick charts
bars = dollar_bars.set_index('date_time')
mpf.plot(bars, type='candle', style='yahoo', volume=True)
```

## Project Structure

```
ticks_data_sampling_preprocessing/
├── pyproject.toml                    # Package configuration
├── uv.lock                          # Dependency lock file
├── bar_generation.ipynb             # Main analysis notebook
├── utils/
│   ├── bar_makers.py                # Core bar generation functions
│   └── threshold_calculators.py    # Threshold calculation utilities
├── ticks_data_ns/                   # Sample tick data
│   └── googl_trade_2023_05_12.txt
└── bars/                            # Generated bar data
    └── dollar_bars.h5
```

## Development

### Running the Analysis

```bash
# Activate the environment
uv shell

# Run the Jupyter notebook
jupyter notebook bar_generation.ipynb
```

### Dependencies

The project uses the following key dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **numba**: JIT compilation for performance
- **scipy**: Statistical functions
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **mplfinance**: Financial charts

## Key Functions

### `get_bars()`
Main function for creating bars from tick data.

**Parameters:**
- `df`: DataFrame with tick data
- `resolution`: Time resolution for time bars ('D', 'H', 'M', 'S')
- `num_units`: Number of time units
- `threshold`: Threshold value for volume/dollar/tick bars
- `batch_size`: Batch size for processing (default: 1e6)
- `bar_type`: Type of bars ('time', 'volume', 'dollar', 'tick')

### Threshold Calculators
- `calculate_dollar_threshold_constant()`: Calculate dollar bar threshold
- `calculate_volume_threshold_constant()`: Calculate volume bar threshold
- `calculate_tick_threshold_constant()`: Calculate tick bar threshold
- `create_metric_threshold_series()`: Create variable thresholds

## Example Results

The library generates bars with the following structure:

| Column | Description |
|--------|-------------|
| date_time | Bar timestamp |
| tick_num | Number of ticks in bar |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| volume | Total volume |
| cum_buy_volume | Cumulative buy volume |
| cum_ticks | Cumulative tick count |
| cum_dollar_value | Cumulative dollar value |

## Performance Benchmarks

- Processes millions of ticks efficiently
- Memory usage scales linearly with batch size
- JIT compilation provides 10-100x speedup over pure Python
- Supports datasets with hundreds of millions of ticks

## Requirements

- Python 3.12+
- See `pyproject.toml` for complete dependency list

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project is inspired by the work on financial data preprocessing and bar generation techniques commonly used in quantitative finance and algorithmic trading.