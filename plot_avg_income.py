import pandas as pd
import matplotlib.pyplot as plt

# Load the data
csv_path = r'VHLSS 2008 Data/avg_log_income_by_age.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.plot(df['age'], df['avg_income'], marker='o')
plt.xlabel('Age')
plt.ylabel('Average Income')
plt.title('Average Income by Age')
plt.grid(True)
plt.tight_layout()
plt.show()
