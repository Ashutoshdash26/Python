file = open("student.txt", "w")

file.write("Name: Ashutosh\n")
file.write("Age: 22\n")
file.write("Course: MCA\n")

file.close()

file = open("student.txt", "r")

print(file.read())
print(file.readline())
file.close()


file = open("student.txt", "a")
file.write("Parth ia a ... \n")
file.close()


file = open("student.txt", "r")
print(file.read())
file.close()

