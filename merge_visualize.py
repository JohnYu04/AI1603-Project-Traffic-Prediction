import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('test_history.csv')
df2 = pd.read_csv('output.csv')

df1['Source'] = 'test_history'
df2['Source'] = 'output'

merged_df = pd.concat([df1, df2])

merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df.sort_values('date', inplace=True)
merged_df.to_csv('merged_data.csv', index=False)

merged_df = merged_df.iloc[169:337]

merged_df.reset_index(drop=True, inplace=True)

plt.figure(figsize=(12, 6))
colors = plt.cm.get_cmap('tab10', 8)  

for i, column in enumerate(merged_df.columns[5:18:3], start=0):
    data_to_plot = merged_df[merged_df['Source'] == 'test_history']
    plt.plot(data_to_plot.index + 1, data_to_plot[column], label=f"{column} (test_history)", color=colors(i))

for i, column in enumerate(merged_df.columns[5:18:3], start=0):
    data_to_plot = merged_df[merged_df['Source'] == 'output']
    plt.scatter(data_to_plot.index + 1, data_to_plot[column], label=f"{column} (output)", color=colors(i), marker='*', s=100)  # 星形标记和增大尺寸

plt.xlabel('Date')
plt.ylabel('Data Values')
plt.title('Data Plot from Merged CSV Files')
plt.legend()
plt.grid(True)
plt.show()
