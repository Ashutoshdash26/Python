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


file = open("student.txt", "a")

file.write("\nCity: Bhubaneswar\n")

file.close()

file = open("student.txt", "r+")
print(file.read())
file.write("Game \n")
file.close()


file = open("student.txt", "a+")

file.write("Game\n")


file.seek(0)


print(file.read())


file.close()


file = open("student.txt", "r")
s1=file.read()
print(s1.find("Game"))
print(s1.count("Game"))
file.close()