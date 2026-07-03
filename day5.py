import csv

data = [
    ["Name", "Age", "City"],
    ["Ashutosh", 22, "Bhubaneswar"],
    ["Parth", 21, "Cuttack"]
]

with open("file.csv", "w",newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)

print("CSV file created successfully!")



with open("file.csv", "r") as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        print(row)





import pandas as pd

# Create data
data = {
    "Employee": ["Rahul", "Sneha", "Aman", "Priya"],
    "Department": ["HR", "IT", "Finance", "Marketing"],
    "Salary": [45000, 70000, 55000, 60000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Parquet
df.to_parquet("employee.parquet", engine="pyarrow", compression="snappy", index=False)

print("Parquet file created successfully!")

# Read the Parquet file
df_read = pd.read_parquet("employee.parquet", engine="pyarrow")

print("\nComplete Data:")
print(df_read)

# Print only Employee column
print("\nEmployees:")
print(df_read["Employee"])