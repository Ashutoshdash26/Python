import csv
with open("study.csv","w",newline="")as file:
    writer=csv.writer(file)

    writer.writerow(['id','Name','age','Course'])
    writer.writerow(["101","Ashutosh","23","MCA"])
    writer.writerow(["19","parth","23","MCA"])

print("CSV file created Successfully ")