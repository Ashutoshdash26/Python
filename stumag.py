class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def grade(self):
        if self.marks >= 90:
            return "A"
        elif self.marks >= 75:
            return "B"
        elif self.marks >= 50:
            return "C"
        else:
            return "Fail"

    def __str__(self):
        return f"{self.name} | Marks: {self.marks} | Grade: {self.grade()}"


students = []

try:
    n = int(input("Enter number of students: "))

    for i in range(n):
        name = input(f"Enter student {i+1} name: ")
        marks = int(input("Enter marks: "))

        if marks < 0 or marks > 100:
            raise ValueError("Marks must be between 0 and 100")

        students.append(Student(name, marks))

except ValueError as e:
    print("Error:", e)

else:
    # Sorting using lambda
    students = sorted(students, key=lambda x: x.marks, reverse=True)

    print("\n--- Student Report ---")

    # List comprehension
    toppers = [s.name for s in students if s.marks >= 90]

    for student in students:
        print(student)

    print("\nToppers:", toppers)

    # File handling
    with open("students.txt", "w") as file:
        for student in students:
            file.write(str(student) + "\n")

    print("\nData saved in students.txt")