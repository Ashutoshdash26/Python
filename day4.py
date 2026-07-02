file = open("student.txt", "w")

file.write("Name: Ashutosh\n")
file.write("Age: 22\n")
file.write("Course: MCA")

file.close()

file = open("student.txt", "r")

print(file.read())
print(file.readline())
print(file.readline())
file.close()