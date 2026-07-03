import pandas as pd

# Create data
data = {
    "Employee": ["Rahul", "Sneha", "Aman", "Priya", "Rohit"],
    "Age": [25, 30, 22, 35, 28],
    "Department": ["HR", "IT", "Finance", "Marketing", "Sales"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Parquet
df.to_parquet("employee.parquet", engine="pyarrow", index=False)

# Read the complete Parquet file
df_read = pd.read_parquet("employee.parquet", engine="pyarrow")

print("Complete Data:")
print(df_read)

# -------------------------------------------------
# Filter 1: Age between 21 and 30
filtered_age = df_read[(df_read["Age"] > 21) & (df_read["Age"] < 30)]

print("\nEmployees with Age between 21 and 30:")
print(filtered_age)

# -------------------------------------------------
# Filter 2: Department is HR or Sales
filtered_department = df_read[
    (df_read["Department"] == "HR") |
    (df_read["Department"] == "Sales")
]

print("\nEmployees in HR or Sales:")
print(filtered_department)

# -------------------------------------------------
# Filter 3: Read Parquet with filters
df_parquet = pd.read_parquet(
    "employee.parquet",
    engine="pyarrow",
    filters=[
        ("Department", "in", ["IT", "Marketing"])
    ]
)

print("\nEmployees in IT or Marketing:")
print(df_parquet)

# -------------------------------------------------
# Filter 4: Age >= 28
filtered_age2 = df_read[df_read["Age"] >= 28]

print("\nEmployees with Age >= 28:")
print(filtered_age2)