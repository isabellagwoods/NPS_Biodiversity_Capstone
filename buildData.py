# Dataset Creation
# Reads downloaded data and outputs dataset

# import packages
import pandas as pd
import os
import glob
# file directories
input = '/Users/isabellawoods/Documents/northwestern 25-26/CAPSTONE/DATA'
output = '/Users/isabellawoods/Documents/northwestern 25-26/CAPSTONE/NPSpecies.csv'

# convert any .xlsx files in the DATA folder to .csv
xlsx_files = glob.glob(os.path.join(input, "*.xlsx"))
for xlsx_file in xlsx_files:
    df_xlsx = pd.read_excel(xlsx_file, engine='calamine')
    park_code = df_xlsx['Park Code'].iloc[0]
    csv_path = os.path.join(input, f"{park_code}.csv")
    df_xlsx.to_csv(csv_path, index=False)
    print(f"Converted: {os.path.basename(xlsx_file)} -> {park_code}.csv")

data_files = glob.glob(os.path.join(input, "*.csv"))

dfs = []
for file in data_files:
    df = pd.read_csv(file)
    dfs.append(df)

# create one big dataset
nps = pd.concat(dfs, ignore_index = True)
nps.to_csv(output, index=False)

# convert CSVs in Traffic Data folder to a single csv
trafficFolder = '/Users/isabellawoods/Documents/northwestern 25-26/CAPSTONE/DATA/Traffic Data'
traffic_files = glob.glob(os.path.join(trafficFolder, "*.csv"))

traffic_list = []
for file in traffic_files:
    traffic_df = pd.read_csv(file)
    traffic_list.append(traffic_df)

traffic = pd.concat(traffic_list, ignore_index = True)
traffic.to_csv('/Users/isabellawoods/Documents/northwestern 25-26/CAPSTONE/DATA/Traffic.csv', index = False)


print(f"Rows: {len(nps)}, Columns: {len(nps.columns)}")
print(f"Files found: {len(data_files)}")
print(nps["Park Code"].unique())
print(traffic.Year.unique())