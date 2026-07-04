import csv
name=input("Enter a name : ")
with open("study.csv","r") as file:
    reader=csv.DictReader(file)
    
    su=False
    for row in reader:
        if(row["Course"].lower()== name.lower()):
            print(row)
            su=True
    if not su:
        print("Record not found ")