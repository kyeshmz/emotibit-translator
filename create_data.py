import numpy as np
from datetime import datetime
import os
import glob
import pandas as pd
import json


def extract_time_ranges(df, first_timestamp):
    """
    Extract first and last 5 minutes of data based on the first timestamp
    """
    # Convert LocalTimestamp to datetime using custom format
    df['datetime'] = pd.to_datetime(df['LocalTimestamp'], unit='s')
    
    # Calculate time ranges
    start_time = pd.to_datetime(first_timestamp, unit='s')
    print('start_time', start_time)
    five_min = pd.Timedelta(minutes=5)
    
    # Get first 5 minutes
    first_5min = df[df['datetime'] <= (start_time + five_min)]
    print('first_5min', first_5min.head())
    print('end of first_5min', first_5min['datetime'].max())
    
    # Get last 5 minutes
    last_5min = df[df['datetime'] >= (df['datetime'].max() - five_min)]
    print('last_5min', last_5min.head())
    print('end of last_5min', last_5min['datetime'].max())
    
    # Return both ranges separately, without the datetime column
    return first_5min.drop('datetime', axis=1), last_5min.drop('datetime', axis=1)


def analyze_sensor_stats(first_5min_df, last_5min_df, sensor_type):
    """
    Calculate statistics for a sensor's first and last 5 minutes
    """
    # Calculate statistics for first 5 minutes
    first_5_std = first_5min_df[sensor_type].std()
    first_5_mean = first_5min_df[sensor_type].mean()
    first_5_median = first_5min_df[sensor_type].median()
    
    # Calculate statistics for last 5 minutes
    last_5_std = last_5min_df[sensor_type].std()
    last_5_mean = last_5min_df[sensor_type].mean()
    last_5_median = last_5min_df[sensor_type].median()
    
    # Calculate statistics for combined data
    combined_df = pd.concat([first_5min_df, last_5min_df])
    combined_std = combined_df[sensor_type].std()
    combined_mean = combined_df[sensor_type].mean()
    combined_median = combined_df[sensor_type].median()
    
    print(f"\nStatistics for {sensor_type}:")
    print(f"First 5 minutes - std: {first_5_std:.4f}, mean: {first_5_mean:.4f}, median: {first_5_median:.4f}")
    print(f"Last 5 minutes - std: {last_5_std:.4f}, mean: {last_5_mean:.4f}, median: {last_5_median:.4f}")
    print(f"Combined - std: {combined_std:.4f}, mean: {combined_mean:.4f}, median: {combined_median:.4f}")
    
    return {
        f'{sensor_type}_first_5_std': first_5_std,
        f'{sensor_type}_first_5_mean': first_5_mean,
        f'{sensor_type}_first_5_median': first_5_median,
        f'{sensor_type}_last_5_std': last_5_std,
        f'{sensor_type}_last_5_mean': last_5_mean,
        f'{sensor_type}_last_5_median': last_5_median,
        f'{sensor_type}_combined_std': combined_std,
        f'{sensor_type}_combined_mean': combined_mean,
        f'{sensor_type}_combined_median': combined_median
    }


def analyze_accelerometer_stats(first_5min_df, last_5min_df):
    """
    Calculate statistics for accelerometer data's first and last 5 minutes
    """
    stats = {}
    
    # Calculate individual axis statistics
    for axis in ['AX', 'AY', 'AZ']:
        first_5_std = first_5min_df[axis].std()
        first_5_mean = first_5min_df[axis].mean()
        first_5_median = first_5min_df[axis].median()
        
        last_5_std = last_5min_df[axis].std()
        last_5_mean = last_5min_df[axis].mean()
        last_5_median = last_5min_df[axis].median()
        
        # Combine first and last 5 minutes for combined stats
        combined_df = pd.concat([first_5min_df, last_5min_df])
        combined_std = combined_df[axis].std()
        combined_mean = combined_df[axis].mean()
        combined_median = combined_df[axis].median()
        
        print(f"\nStatistics for {axis}:")
        print(f"First 5 minutes - std: {first_5_std:.4f}, mean: {first_5_mean:.4f}, median: {first_5_median:.4f}")
        print(f"Last 5 minutes - std: {last_5_std:.4f}, mean: {last_5_mean:.4f}, median: {last_5_median:.4f}")
        print(f"Combined - std: {combined_std:.4f}, mean: {combined_mean:.4f}, median: {combined_median:.4f}")
        
        stats.update({
            f'{axis}_first_5_std': first_5_std,
            f'{axis}_first_5_mean': first_5_mean,
            f'{axis}_first_5_median': first_5_median,
            f'{axis}_last_5_std': last_5_std,
            f'{axis}_last_5_mean': last_5_mean,
            f'{axis}_last_5_median': last_5_median,
            f'{axis}_combined_std': combined_std,
            f'{axis}_combined_mean': combined_mean,
            f'{axis}_combined_median': combined_median
        })
    
    # Calculate combined accelerometer statistics (all axes together)
    first_5_combined = np.sqrt(first_5min_df['AX']**2 + first_5min_df['AY']**2 + first_5min_df['AZ']**2)
    last_5_combined = np.sqrt(last_5min_df['AX']**2 + last_5min_df['AY']**2 + last_5min_df['AZ']**2)
    
    first_5_total_std = first_5_combined.std()
    first_5_total_mean = first_5_combined.mean()
    first_5_total_median = first_5_combined.median()
    
    last_5_total_std = last_5_combined.std()
    last_5_total_mean = last_5_combined.mean()
    last_5_total_median = last_5_combined.median()
    
    # Combine first and last 5 minutes
    all_combined = pd.concat([first_5_combined, last_5_combined])
    combined_total_std = all_combined.std()
    combined_total_mean = all_combined.mean()
    combined_total_median = all_combined.median()
    
    print(f"\nStatistics for combined accelerometer magnitude:")
    print(f"First 5 minutes - std: {first_5_total_std:.4f}, mean: {first_5_total_mean:.4f}, median: {first_5_total_median:.4f}")
    print(f"Last 5 minutes - std: {last_5_total_std:.4f}, mean: {last_5_total_mean:.4f}, median: {last_5_total_median:.4f}")
    print(f"Combined - std: {combined_total_std:.4f}, mean: {combined_total_mean:.4f}, median: {combined_total_median:.4f}")
    
    stats.update({
        'accel_magnitude_first_5_std': first_5_total_std,
        'accel_magnitude_first_5_mean': first_5_total_mean,
        'accel_magnitude_first_5_median': first_5_total_median,
        'accel_magnitude_last_5_std': last_5_total_std,
        'accel_magnitude_last_5_mean': last_5_total_mean,
        'accel_magnitude_last_5_median': last_5_total_median,
        'accel_magnitude_combined_std': combined_total_std,
        'accel_magnitude_combined_mean': combined_total_mean,
        'accel_magnitude_combined_median': combined_total_median
    })
    
    return stats


def main():
    # Get all CSV files in kye_run_1 directory
    data_dir = 'nathan_run_2'
    data_files = glob.glob(f'{data_dir}/*.csv')
    prefix = f'{data_dir}_'  # Prefix for all output files
    
    # Dictionary to store dataframes
    all_data = {}
    all_data_ranges = {}
    
    # Track the first timestamp we see
    first_timestamp = None
    
    # Create ranges directory if it doesn't exist
    ranges_dir = 'ranges'
    if not os.path.exists(ranges_dir):
        os.makedirs(ranges_dir)
    
    # Dictionary to store statistics
    all_stats = {}
    
    # Dictionary to store accelerometer data
    accel_first_5 = None
    accel_last_5 = None
    
    for file in data_files:
        if not ('HR' in file or 'T1' in file or 'EA' in file or 'TH' in file or 'BI' in file or 
                'AX' in file or 'AY' in file or 'AZ' in file):
            continue
            
        print(f"\nReading file: {file}")
        try:
            df = pd.read_csv(file)
            
            # Get the type from filename
            if 'HR' in file:
                data_type = 'HR'
            elif 'T1' in file:
                data_type = 'T1'
            elif 'EA' in file:
                data_type = 'EA'
            elif 'TH' in file:
                data_type = 'TH'
            elif 'BI' in file:
                data_type = 'BI'
            elif 'AX' in file:
                data_type = 'AX'
            elif 'AY' in file:
                data_type = 'AY'
            elif 'AZ' in file:
                data_type = 'AZ'
            
            # Keep LocalTimestamp and the sensor data
            df = df[['LocalTimestamp', data_type]]
            
            # Store the first timestamp we see
            if first_timestamp is None:
                first_timestamp = df['LocalTimestamp'].iloc[0]
            
            all_data[data_type] = df
            
            # Extract ranges for individual sensor
            first_5min, last_5min = extract_time_ranges(df.copy(), first_timestamp)
            
            if data_type in ['AX', 'AY', 'AZ']:
                # For accelerometer data, combine into single files
                if accel_first_5 is None:
                    accel_first_5 = first_5min
                    accel_last_5 = last_5min
                else:
                    # Merge with existing accelerometer data
                    accel_first_5 = pd.merge(accel_first_5, first_5min, on='LocalTimestamp', how='outer')
                    accel_last_5 = pd.merge(accel_last_5, last_5min, on='LocalTimestamp', how='outer')
            else:
                # Save individual files for non-accelerometer data
                first_5min_filename = os.path.join(ranges_dir, f'{prefix}{data_type}_first_5.csv')
                first_5min.to_csv(first_5min_filename, index=False)
                print(f"Saved first 5 minutes for {data_type} to {first_5min_filename}")
                
                last_5min_filename = os.path.join(ranges_dir, f'{prefix}{data_type}_last_5.csv')
                last_5min.to_csv(last_5min_filename, index=False)
                print(f"Saved last 5 minutes for {data_type} to {last_5min_filename}")
            
            # Calculate statistics for HR, T1, EA, TH, and BI
            if data_type in ['HR', 'T1', 'EA', 'TH', 'BI']:
                stats = analyze_sensor_stats(first_5min, last_5min, data_type)
                all_stats.update(stats)
            
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
    
    # Save combined accelerometer data
    if accel_first_5 is not None:
        # Sort by timestamp
        accel_first_5 = accel_first_5.sort_values('LocalTimestamp')
        accel_last_5 = accel_last_5.sort_values('LocalTimestamp')
        
        # Calculate accelerometer statistics
        accel_stats = analyze_accelerometer_stats(accel_first_5, accel_last_5)
        all_stats.update(accel_stats)
        
        # Save combined accelerometer files
        accel_first_filename = os.path.join(ranges_dir, f'{prefix}accelerometer_first_5.csv')
        accel_first_5.to_csv(accel_first_filename, index=False)
        print(f"\nSaved combined accelerometer first 5 minutes to {accel_first_filename}")
        
        accel_last_filename = os.path.join(ranges_dir, f'{prefix}accelerometer_last_5.csv')
        accel_last_5.to_csv(accel_last_filename, index=False)
        print(f"\nSaved combined accelerometer last 5 minutes to {accel_last_filename}")
    
    if all_data:
        # Process full dataset
        combined_df = list(all_data.values())[0]
        
        # Merge all other dataframes on LocalTimestamp
        for df in list(all_data.values())[1:]:
            # Ensure we're only merging on LocalTimestamp
            combined_df = pd.merge(combined_df, df, on='LocalTimestamp', how='outer')
        
        # Sort by LocalTimestamp
        combined_df = combined_df.sort_values('LocalTimestamp')
        
        # Save combined data
        output_file = f'{prefix}combined_sensor_data.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved combined data to {output_file}")
        
        # Process time ranges dataset for combined data
        first_5min_combined, last_5min_combined = extract_time_ranges(combined_df.copy(), first_timestamp)
        
        # Save combined first 5 minutes
        first_5min_output = os.path.join(ranges_dir, f'{prefix}combined_first_5.csv')
        first_5min_combined.to_csv(first_5min_output, index=False)
        print(f"\nSaved combined first 5 minutes to {first_5min_output}")
        
        # Save combined last 5 minutes
        last_5min_output = os.path.join(ranges_dir, f'{prefix}combined_last_5.csv')
        last_5min_combined.to_csv(last_5min_output, index=False)
        print(f"\nSaved combined last 5 minutes to {last_5min_output}")
        
        # Display summary
        print(f"\nTotal rows in full dataset: {len(combined_df)}")
        print(f"Total rows in first 5 minutes: {len(first_5min_combined)}")
        print(f"Total rows in last 5 minutes: {len(last_5min_combined)}")
        print(f"Columns: {combined_df.columns.tolist()}")
        
        # Save statistics to JSON file
        if all_stats:
            stats_file = os.path.join('ranges', f'{prefix}range_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
            print(f"\nSaved statistics to {stats_file}")
        
    else:
        print("No data was processed successfully")


if __name__ == "__main__":
    main()
