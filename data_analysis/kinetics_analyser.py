import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

def extract_kinetics(time: np.ndarray, 
                    intensity: np.ndarray, 
                    channel_name: str = "",
                    time_window: Optional[Tuple[float, float]] = None,
                    exclude_outliers: bool = True,
                    outlier_threshold: float = 3.0) -> Dict:
    """
    Extract kinetic parameters from nephelometry time series data.
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds)
    intensity : array-like
        Normalized intensity values (should start around 1.0)
    channel_name : str
        Name of the spectral channel (e.g., 'F3', 'Clear')
    time_window : tuple, optional
        (start_time, end_time) to restrict analysis window
    exclude_outliers : bool
        Whether to exclude statistical outliers (useful for F8 spikes)
    outlier_threshold : float
        Z-score threshold for outlier detection
        
    Returns:
    --------
    dict : Dictionary containing kinetic parameters
    """
    
    # Convert to numpy arrays
    time = np.array(time)
    intensity = np.array(intensity)
    
    # Apply time window if specified
    if time_window is not None:
        mask = (time >= time_window[0]) & (time <= time_window[1])
        time = time[mask]
        intensity = intensity[mask]
    
    # Remove outliers if requested (useful for F8 channel)
    if exclude_outliers and len(intensity) > 10:
        z_scores = np.abs(stats.zscore(intensity))
        outlier_mask = z_scores < outlier_threshold
        time = time[outlier_mask]
        intensity = intensity[outlier_mask]
    
    if len(time) < 5:
        return {"error": "Insufficient data points after filtering"}
    
    # Initialize results dictionary
    results = {
        "channel": channel_name,
        "n_points": len(time),
        "time_range": (time.min(), time.max()),
        "duration": time.max() - time.min()
    }
    
    # Basic statistics
    results["initial_value"] = intensity[0]
    results["final_value"] = intensity[-1]
    results["total_change"] = intensity[-1] - intensity[0]
    results["percent_change"] = ((intensity[-1] - intensity[0]) / intensity[0]) * 100
    results["mean_intensity"] = np.mean(intensity)
    results["std_intensity"] = np.std(intensity)
    results["cv_percent"] = (results["std_intensity"] / results["mean_intensity"]) * 100
    
    # Linear trend analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(time, intensity)
    results["linear_slope"] = slope
    results["linear_slope_per_min"] = slope * 60  # Convert to per minute
    results["linear_r_squared"] = r_value**2
    results["linear_p_value"] = p_value
    results["linear_std_error"] = std_err
    
    # Exponential fitting (for settling kinetics)
    try:
        def exponential_model(t, a, b, c):
            """Exponential model: y = a * exp(-b * t) + c"""
            return a * np.exp(-b * t) + c
        
        # Initial guess
        p0 = [intensity[0] - intensity[-1], 0.01, intensity[-1]]
        
        popt, pcov = curve_fit(exponential_model, time, intensity, p0=p0, maxfev=1000)
        
        results["exp_amplitude"] = popt[0]
        results["exp_rate_constant"] = popt[1]
        results["exp_offset"] = popt[2]
        results["exp_half_time"] = np.log(2) / popt[1] if popt[1] > 0 else np.inf
        
        # Calculate R² for exponential fit
        y_pred = exponential_model(time, *popt)
        ss_res = np.sum((intensity - y_pred) ** 2)
        ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
        results["exp_r_squared"] = 1 - (ss_res / ss_tot)
        
    except Exception as e:
        results["exp_fit_error"] = str(e)
    
    # Rate analysis (derivative-based)
    if len(time) > 3:
        # Calculate numerical derivative
        dt = np.diff(time)
        dy = np.diff(intensity)
        rates = dy / dt
        
        results["max_rate"] = np.max(rates)
        results["min_rate"] = np.min(rates)
        results["mean_rate"] = np.mean(rates)
        results["initial_rate"] = rates[0] if len(rates) > 0 else 0
        results["final_rate"] = rates[-1] if len(rates) > 0 else 0
        
        # Find time of maximum rate change
        max_rate_idx = np.argmax(np.abs(rates))
        results["time_max_rate"] = time[max_rate_idx]
    
    # Stability analysis
    # Calculate moving standard deviation to assess stability
    if len(intensity) > 10:
        window_size = min(10, len(intensity) // 3)
        moving_std = []
        for i in range(len(intensity) - window_size + 1):
            window_std = np.std(intensity[i:i + window_size])
            moving_std.append(window_std)
        
        results["stability_metric"] = np.mean(moving_std)
        results["early_stability"] = np.std(intensity[:len(intensity)//3])
        results["late_stability"] = np.std(intensity[2*len(intensity)//3:])
    
    return results

def analyze_all_channels(df: pd.DataFrame, 
                        time_col: str = 'time',
                        **kwargs) -> pd.DataFrame:
    """
    Analyze kinetics for all channels in a dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time and channel columns
    time_col : str
        Name of time column
    **kwargs : additional arguments passed to extract_kinetics
    
    Returns:
    --------
    pandas.DataFrame : Results for all channels
    """
    channels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']	
    clear_values = df['raw_Clear']

    results = []
    
    for channel in channels:
        column_name = f'raw_{channel}'

        if column_name in df.columns:
            normalized_on_clear_values = df[column_name] / clear_values
            kinetics = extract_kinetics(df[time_col], normalized_on_clear_values, channel, **kwargs)
            results.append(kinetics)
    
    return pd.DataFrame(results)

def plot_kinetics_summary(csv_file, time_window: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create summary plots of kinetic analysis results.
    """
    channels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Remove the last row if it contains any NaN values
    if df.iloc[-1].isna().any():
        df = df.iloc[:-1]
        print("Removed last row due to NaN values")    

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Convert to seconds
    start_time = df['timestamp'].iloc[0]
    df['time_seconds'] = df['timestamp'] - start_time        

    # Analyze kinetics
    kinetics_df = analyze_all_channels(df, 'time_seconds', time_window=time_window)
    
    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    # Extract filename without path and extension for title
    filename = os.path.basename(csv_file)
    title = os.path.splitext(filename)[0]
 
    # Plot 1: Linear slopes
    if 'linear_slope_per_min' in kinetics_df.columns:
        ax = axes[0]
        bars = ax.bar(kinetics_df['channel'], kinetics_df['linear_slope_per_min'])
        ax.set_title('Linear Slope (per minute)')
        ax.set_ylabel('Slope (/min)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Color bars by sign
        for bar, slope in zip(bars, kinetics_df['linear_slope_per_min']):
            bar.set_color('red' if slope > 0 else 'blue')
    
    # Plot 2: R² values for linear fits
    if 'linear_r_squared' in kinetics_df.columns:
        ax = axes[1]
        ax.bar(kinetics_df['channel'], kinetics_df['linear_r_squared'])
        ax.set_title('Linear Fit Quality (R²)')
        ax.set_ylabel('R²')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Total percent change
    if 'percent_change' in kinetics_df.columns:
        ax = axes[2]
        bars = ax.bar(kinetics_df['channel'], kinetics_df['percent_change'])
        ax.set_title('Total Percent Change')
        ax.set_ylabel('Change (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Color bars by sign
        for bar, change in zip(bars, kinetics_df['percent_change']):
            bar.set_color('red' if change > 0 else 'blue')
    
    # Plot 4: Stability metric
    if 'stability_metric' in kinetics_df.columns:
        ax = axes[3]
        ax.bar(kinetics_df['channel'], kinetics_df['stability_metric'])
        ax.set_title('Stability Metric (lower = more stable)')
        ax.set_ylabel('Moving Std Dev')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 5: Exponential half-times (if available)
    max_half_time = time_window[-1] - time_window[0] if time_window else 60  # Default to 60 seconds if no window specified
    if 'exp_half_time' in kinetics_df.columns:
        ax = axes[4]
        # Filter out infinite values
        finite_mask = np.isfinite(kinetics_df['exp_half_time'])
        channels_finite = kinetics_df.loc[finite_mask, 'channel']
        half_times_finite = kinetics_df.loc[finite_mask, 'exp_half_time']
        
        if len(half_times_finite) > 0:
            ax.bar(channels_finite, half_times_finite)
            ax.set_title('Exponential Half-Times')
            ax.set_ylabel('Half-time (s)')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, max_half_time)

    # Plot 6: Initial vs Final values
    if 'initial_value' in kinetics_df.columns and 'final_value' in kinetics_df.columns:
        ax = axes[5]
        ax.scatter(kinetics_df['initial_value'], kinetics_df['final_value'])
        
        # Add channel labels
        for i, channel in enumerate(kinetics_df['channel']):
            ax.annotate(channel, 
                       (kinetics_df['initial_value'].iloc[i], kinetics_df['final_value'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line
        min_val = min(kinetics_df['initial_value'].min(), kinetics_df['final_value'].min())
        max_val = max(kinetics_df['initial_value'].max(), kinetics_df['final_value'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Initial Value')
        ax.set_ylabel('Final Value')
        ax.set_title('Initial vs Final Values')

    # After creating your subplots and before plt.tight_layout()
    plt.suptitle(title, fontsize=16)
   
    plt.tight_layout()
    return fig, kinetics_df

# Example usage and comparison function
def compare_conditions(condition1_df: pd.DataFrame, 
                      condition2_df: pd.DataFrame,
                      condition1_name: str = "Condition 1",
                      condition2_name: str = "Condition 2",
                      time_col: str = 'time') -> pd.DataFrame:
    """
    Compare kinetics between two different experimental conditions.
    """
    
    # Analyze both conditions
    kinetics1 = analyze_all_channels(condition1_df, time_col)
    kinetics2 = analyze_all_channels(condition2_df, time_col)
    
    # Merge and compare
    comparison = kinetics1.set_index('channel').join(
        kinetics2.set_index('channel'), 
        rsuffix=f'_{condition2_name}', 
        lsuffix=f'_{condition1_name}'
    )
    
    # Calculate differences in key metrics
    if f'linear_slope_per_min_{condition1_name}' in comparison.columns:
        comparison['slope_difference'] = (
            comparison[f'linear_slope_per_min_{condition2_name}'] - 
            comparison[f'linear_slope_per_min_{condition1_name}']
        )
    
    if f'percent_change_{condition1_name}' in comparison.columns:
        comparison['percent_change_difference'] = (
            comparison[f'percent_change_{condition2_name}'] - 
            comparison[f'percent_change_{condition1_name}']
        )
    
    return comparison
