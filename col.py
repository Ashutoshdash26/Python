from abc import ABC, abstractmethod
from functools import wraps
import time

# ==========================
# Decorator
# ==========================
def execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution Time: {end - start:.6f} seconds")
        return result
    return wrapper

# ==========================
# Abstract Class
# ==========================
class Employee(ABC):

    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary  # Encapsulation

    @property
    def salary(self):
        return self.__salary

    @abstractmethod
    def calculate_bonus(self):
        pass

# ==========================
# Inheritance & Polymorphism
# ==========================
class Developer(Employee):

    def calculate_bonus(self):
        return self.salary * 0.20

class Manager(Employee):

    def calculate_bonus(self):
        return self.salary * 0.30

# ==========================
# Generator
# ==========================
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# ==========================
# Custom Iterator
# ==========================
class CountDown:

    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# ==========================
# Exception Handling
# ==========================
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"
    finally:
        print("Division Attempt Completed")

# ==========================
# List Comprehension
# ==========================
squares = [x*x for x in range(10)]

# ==========================
# Lambda Function
# ==========================
numbers = [5, 1, 8, 2]
sorted_numbers = sorted(numbers, key=lambda x: x)

# ==========================
# Main Function
# ==========================
@execution_time
def main():

    employees = [
        Developer("Ashutosh", 50000),
        Manager("John", 80000)
    ]

    print("=== Employee Bonuses ===")
    for emp in employees:
        print(emp.name, emp.calculate_bonus())

    print("\n=== Fibonacci Generator ===")
    for num in fibonacci(10):
        print(num, end=" ")

    print("\n\n=== Countdown Iterator ===")
    for i in CountDown(5):
        print(i, end=" ")

    print("\n\n=== Exception Handling ===")
    print(divide(10, 0))

    print("\n=== List Comprehension ===")
    print(squares)

    print("\n=== Lambda Sorting ===")
    print(sorted_numbers)

if __name__ == "__main__":
    main()