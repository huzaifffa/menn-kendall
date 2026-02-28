
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def sens_slope(data):
    n = len(data)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope = (data[j] - data[i]) / (j - i)
            slopes.append(slope)
    return np.median(slopes)

def analyze_location(df, location, parameters, years):
    print(f"\nLocation: {location}")
    for param in parameters:
        if param not in df.columns:
            print(f"Parameter {param} not found in data.")
            continue
        data = pd.to_numeric(df[param], errors='coerce').dropna().values
        if len(data) == 0:
            print(f"No data for {param} in {location}.")
            continue
        slope = sens_slope(data)
        tau, p_value = kendalltau(years[:len(data)], data)
        print(f"  {param}: Sen's Slope = {slope:.4f}, p-value = {p_value:.4g}")
        plt.figure()
        plt.plot(years[:len(data)], data, marker='o')
        plt.title(f'{location} - {param} (1901-2024)')
        plt.xlabel('Year')
        plt.ylabel(param)
        plt.grid(True)
        plt.savefig(f'{location}_{param}_trend.png')
        plt.close()

def main():
    df = pd.read_excel('mennkendall.xlsx', engine='openpyxl')
    # List of locations and parameters
    locations = ['Badin', 'Dadu', 'Gotki', 'Kashmore', 'Shahdadkot', 'Shaheed Benazirabad', 'Sanghar', 'Thar']
    parameters = ['temperature', 'precipitation', 'vapor pressure']
    years = np.arange(1901, 2025)
    # Assume each location is a separate sheet or column group
    # If each location is a sheet:
    if hasattr(df, 'sheet_names'):
        for location in locations:
            if location in df.sheet_names:
                loc_df = pd.read_excel('mennkendall.xlsx', sheet_name=location, engine='openpyxl')
                analyze_location(loc_df, location, parameters, years)
    else:
        # If columns are like Badin_temperature, Badin_precipitation, etc.
        for location in locations:
            loc_data = {}
            for param in parameters:
                col_name = f'{location}_{param}'
                if col_name in df.columns:
                    loc_data[param] = df[col_name]
            if loc_data:
                loc_df = pd.DataFrame(loc_data)
                analyze_location(loc_df, location, parameters, years)

if __name__ == "__main__":
    main()
