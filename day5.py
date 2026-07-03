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