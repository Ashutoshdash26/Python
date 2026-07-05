import csv

# 1. Writing to the CSV file
with open("study.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'Name', 'age', 'Course'])
    writer.writerow(["101", "Ashutosh", "23", "MCA"])
    writer.writerow(["19", "parth", "23", "MCA"])

print("CSV file created Successfully\n")


# 2. Reading from the CSV file
with open("study.csv", "r") as file:
    # --- Using standard csv.reader ---
    print("--- Reading with csv.reader (Lists) ---")
    reader = csv.reader(file)
    for i in reader:
        print(i)
        
    print("\n" + "-"*40 + "\n") # Visual separator

    # --- Resetting the file pointer ---
    file.seek(0) 

    # --- Using csv.DictReader ---
    print("--- Reading with csv.DictReader (Dictionaries) ---")
    re = csv.DictReader(file)
    for i in re:
        print(i) # Wrapped in dict() for clean printing


data=[
    ["34","Mimi","54","BCA"]
]



# 1. Prepare multiple new rows of data to add
new_data = [
    ["34", "Mimi", "54", "BCA"],
    ["102", "Rahul", "21", "BTech"],
    ["55", "Sara", "22", "BSc"]
]

# 2. Append the new data to the CSV file
with open("study.csv", "a", newline="") as file:
    append = csv.writer(file)
    # Use writerows (plural) because new_data is a list of lists
    append.writerows(new_data) 

print("Data appended successfully!\n")


# 3. Open the file separately in read mode to view the final results
print("--- Current CSV Contents ---")
with open("study.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
  





print("---------------------------------------------------------------------")

with open("study.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        # {} {} {} {} creates four placeholders separated by tabs (\t)
        print("{:<5} {:<13} {:<5} {:<5}".format(*row))


    file.seek(0)


import csv
name=input("Enter a name : ")
with open("study.csv","r") as file:
    reader=csv.DictReader(file)
    
    su=False
    for row in reader:
        if(row["Name"].lower()== name.lower()):
            print(row)
            su=True
    if not su:
        print("Record not found ")




import csv

rows = []
# 1. Rename 'input' variable so it doesn't conflict with Python's input() function
search_id = input("Enter the id you are searching for: ")

with open("study.csv", "r+", newline="") as file: 
    reader = csv.reader(file)
    
    # Read everything into memory first
    all_rows = list(reader)
    
    for row in all_rows:
        if row: # Ensure the row isn't empty
            # 2. Compare string to string (search_id is a string)
            if row[0] == search_id: 
                new_id = input(f"ID {search_id} found! Enter the new ID you want to change it to: ")
                row[0] = new_id  # Update the ID column (index 0)
                
            rows.append(row) # Keep track of all rows (modified or not)
            
    # 3. Rewrite the file cleanly using r+ mechanisms
    file.seek(0)
    writer = csv.writer(file)
    writer.writerows(rows)
    file.truncate() # Chop off any leftover old data

print("Update Successful!")



print("___________________________________________________")

