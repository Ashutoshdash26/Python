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

print("____________________________________________")






# 1. Create the DataFrame
data = {
    "Product": ["Laptop", "Mouse", "Monitor"],
    "Price": [1200.50, 25.00, 300.00],
    "Stock": [15, 120, 45]
}
df = pd.DataFrame(data)

# # 2. Write DataFrame to a Parquet file
# # 'engine' defaults to 'auto' but specifying 'pyarrow' ensures explicit behavior
df.to_parquet("inventory.parquet", engine="pyarrow", compression="snappy", index=False)
print("File written successfully!")

# # 3. Read Parquet file back into a DataFrame
df_read = pd.read_parquet("inventory.parquet", engine="pyarrow")

# # Display the data
print("\nRead DataFrame:")
print(df_read)
print(df_read["Product"])

