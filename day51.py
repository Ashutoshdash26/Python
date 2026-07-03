import pandas as pd

data = {
    "name": ["Ashutosh", "Parth", "Bis"],
    "age": [22, 21, 19],
    "city": ["BBSR", "Cuttack", "BBSR"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("demo_csv.csv", index=False)

# Read CSV
df2 = pd.read_csv("demo_csv.csv")
print(df2)

print("--------------------------")

# Save to Parquet
df.to_parquet("abc.parquet", index=False)

# Read Parquet
data = pd.read_parquet("abc.parquet")

# Filter rows where age > 20
result = data[data["age"] > 20]

print(result)