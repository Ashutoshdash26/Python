# # 29/6/2026-----------------------------------------------------------------
# day 1


# print("Ashutosh Dash")
# print("Jupyter working")
# print("game of thrones "," you are a grate seris")
# from math import *
# from datetime import datetime as dd
# print(dd.now())
# print(sqrt(16))
# print(pi)
# print(pow(2,3))

# x=5
# y=10
# def z(x,y):
#     return y/x
# print(x+y) # addition operation, hence it will return a value
# print(x*y) # multiplication operation, hence it will return a value
# print(True) # built-in constant value
# print((x,y)) # tuple
# print(z(x,y)) # function call

# def myfuction(name="Ashutoh Dash"):
#     print(f"My name is {name}")
# print("out")
# myfuction()
# myfuction("Parth")
# myemplist=[100,"Abhi",99.99]
# print(myemplist)
# print(type(myemplist))

# cources=("java ","python ","c++","java script","python ")
# print(cources.count("python "))
# print(cources.index("python "))


# de={10,30,20,60,50,70}
# de.add(40)
# print(de)

# mylo=[
#     {"AA","BB","Cc"},
#     {34,54,67}
# ]
# print(mylo)


# mydi={"empid":1091,"empname":" Ashutosh"}
# print(mydi)
# print(mydi.keys())
# print(mydi.values())

# empid=input("Enter employ id ")
# empname=input("Enter employ name ")
# empsalary=input("Enter employ salary ")


# print(type(10))
# a=15
# if(a>10):
#    print("game")

# for i in range(1,5):
#     print(i)


# Day 2------------------------------------------------------------------------
#30/6/2026



# a=2+5j
# b=3+3j
# print(a+b)
# print(id)
# help(id)
# print(bin(20))
# b=bin(10)
# a = 10
# print(a << 2)
# print(bin(a << 2))
# x=150
# y=100
# print(x&y)
# print(x|y)

# x = 10
# y = 20

# print(x > 5 and y > 15)

# age = 16

# print(age >= 18 or age == 16)

# print(True ^ False)


#  Interactive Number Converter based 

# print("--- Number System Converter ---")

# while True:
#     user_input = input("\nEnter a decimal integer (or type 'exit' to quit): ").strip()
    
    

#     if user_input.lower() == 'exit':
#         print("Exiting program. Goodbye!")
#         break
        
#     if not user_input:
#         continue
        
   
#     if not user_input.isdigit() and not (user_input.startswith('-') and user_input[1:].isdigit()):
#         print("Invalid input. Please enter a valid integer.")
#         continue

#     x = int(user_input)
    
#     print(f"\nResults for decimal x = {x}:")
#     print(f"  • Octal representation:       {oct(x)}")
#     print(f"  • Hexadecimal representation: {hex(x)}")
#     print(f"  • Binary representation:      {bin(x)}")
    
    
#     binary_str = bin(x).replace("0b", "").replace("-", "")
#     has_high_bits = False
    
#     for bit in binary_str:
#         if bit == '1':
#             has_high_bits = True
#             break  
            
#     if has_high_bits:
#         print("  • Note: This number contains active high (1) bits.")
#     else:
#         print("  • Note: This number represents absolute zero (no active high bits).")
# def add(a=101,b=109):
#     return a+b
# print(add())
# print(add(100,300))


# def find_factorial(number):
#     if number <= 1:
#         return 1
#     else:
#         return number * find_factorial(number - 1)

# n = int(input("Enter a number: "))
# result = find_factorial(n)

# print("Factorial =", result)

# def adder(*num):
#     sum=0
#     for n in num:
#         sum=sum+n
#     print("Sum : ",sum)
# adder()
# adder(10,20,30,40)

# def myfun(arg1, *argv):
#     print("First Argument:", arg1)

#     for arg in argv:
#         print("Argument:", arg)

#     print("#" * 20)
#     print("The argv:", argv)
#     print("#" * 20)

# myfun("Hello", "to", "Python", "Program")





# def perform(**kwargs):
#     print(kwargs)
#     print(type(kwargs))

# perform(banana=5, mango=10, cherry=4)



# def perform(a, b, **kwargs):
#     print(kwargs)
#     if kwargs['action'] == 'mul':
#         return a * b
#     else:
#         return a + b


# print(perform(20, 15, action='aaa'))

# print(perform(20, 15, action='mul'))




#day 3-------------------------------------------------------------------------
#1/07/2026


# f=lambda x:x*x
# print(f(8))

# g=lambda x,y,z:x**y+z
# print(g(5,6,7))
# check_age=lambda age:"Adult" if age>=18 else "Minor"
# print(check_age(25))


# #map filter reduce

# l1=[3,4,5,7.3,9]
# print(list(map(lambda x:x*x,l1)))
# print(tuple(map(lambda x:x*x,l1)))
# print(set(map(lambda x:x*x,l1)))


# print(list(filter(lambda x:x%2==0,l1)))

# from functools import reduce
# print(reduce(lambda x,y:x*y,l1))


# l2=[20,40,10,40,25,19,11]
# l2=list(filter(lambda x:x%2!=0,l2))
# l2=list((map(lambda x:x*x,l2)))
# print(reduce(lambda x,y:x+y,l2))
# print(l2)

# from functools import reduce

# l2 = [20, 40, 10, 40, 25, 19, 11]

# result = reduce(
#     lambda x, y: x + y,
#     map(
#         lambda x: x * x,
#         filter(lambda x: x % 2 != 0, l2)
#     )
# )

# print(result)

# --- Part 1: any and all ---
# boolean_list = [True, True, False]

# # Check if all elements are true
# result1 = all(boolean_list)
# print("all will return ==>", result1)

# result2 = any(boolean_list)
# print("any will return ==>", result2)


# # --- Part 2: enumerate ---
# languages = ['Python', 'Java', 'JavaScript']
# enumerate_prime = enumerate(languages)

# # Convert enumerate object to list
# print(list(enumerate_prime))



# # --- Part 3: abs function ---
# x = -200
# print(abs(x))



# languages = ['Python', 'Java', 'JavaScript']

# for index, language in enumerate(languages, start=1):
#     print(index, language)


# fruits = ("Apple", "Banana", "Mango")

# for i, fruit in enumerate(fruits):
#     print(i, fruit)

# word = "Python"

# for index, letter in enumerate(word):
#     print(index, letter)


# --- Part 1: Using the reversed() function on different sequences ---

# For string
# seq_string = 'Python'
# print(list(reversed(seq_string)))

# # For tuple
# seq_tuple = ('P', 'y', 't', 'h', 'o', 'n')
# print(list(reversed(seq_tuple)))

# # For range
# seq_range = range(5, 9)
# print(list(reversed(seq_range)))

# # For list
# seq_list = [1, 2, 4, 3, 5]
# print(list(reversed(seq_list)))


# # --- Part 2: Sorting a list using sorted() ---

# numbers = [4, 2, 12, 8]
# sorted_numbers = sorted(numbers)
# print(sorted_numbers)


# # --- Part 3: Creating lists (Partially visible at the bottom) ---

# languages = ['Java', 'Python', 'JavaScript', 'C++', 'Scala']
# versions = [14, 3, 6]

# # Printing them out to complete the snippet logically
# print(languages)
# print(versions)

# numbers = [10, 20, 30, 40, 50]

# print(list(reversed(numbers)))

# def make_pretty(func):
#     def inner():
#         print("I got decorated")
#         func()
#     return inner

# #makepretty(ordinary)
# @make_pretty
# def ordinary():
#     print("I am ordinary")

# ordinary()




# def welcome(func):
#     def inner():
#         print("Welcome!")
#         func()
#         print("Thank you!")
#     return inner

# @welcome
# def greet():
#     print("Hello")

# greet()



# def login_required(func):
#     def inner():
#         password = input("Enter Password: ")

#         if password == "admin123":
#             func()
#         else:
#             print("Access Denied!")

#     return inner

# @login_required
# def profile():
#     print("Welcome to your Profile")

# profile()



# Using a loop to generate a list of square numbers
# squr = []
# for x in range(1, 11):
#     sq = x**2
#     squr.append(sq)
# print("Using For : ", squr)

# # Using a comprehension to generate a list of square numbers
# squr = [x**2 for x in range(1, 11)]
# print("Using List Comprehension : ", squr)



# # Using a loop to convert a list of names to upper case
# colors = ['Red', 'Blue', 'Green', 'Black', 'White']
# upper_cols = []
# for cols in colors:
#     upper_cols.append(cols.upper())

# print("Using Normal method : ", upper_cols)






# # Using a comprehension to convert a list of names to upper case
# colors = ['Red', 'Blue', 'Green', 'Black', 'White']
# upper_cols = [cols.upper() for cols in colors]
# print("Using List Comprehension : ", upper_cols)



## Exception handling
# try
# except
# raise
# finally

try:
    num = int(input("Enter a number: "))
    print(10 / num)

except ZeroDivisionError:
    print("Division by zero is not allowed.")

except ValueError:
    print("Please enter a valid integer.")



try:
    age = int(input("Enter your age: "))

    if age < 18:
        raise ValueError("Age must be at least 18.")

except ValueError as e:
    print("Error:", e)

else:
    print("Eligible")

finally:
    print("Thank You!")



x = 10
y = 0
l1 = [30,20,10,40,50]

res = x + y
print(f"The sum of {x} and {y} is {res}")

res = x * y
print(f"The prod of {x} and {y} is {res}")

try:
    res = z / y
    print(f"The div of {x} and {y} is {res}")
except (ZeroDivisionError,IndexError) :
    print("ZDE / IE is handled")
except :
    print("All other exception will be handled")

res = x - y
print(f"The sub of {x} and {y} is {res}")



def withdraw(balance, amount):
    try:
        if amount > balance:
            raise ValueError("Insufficient Balance")

        balance -= amount
        print("Remaining Balance:", balance)

    except ValueError as e:
        print(e)

    finally:
        print("Transaction Completed")

withdraw(5000, 2000)


class MyError(Exception):
    pass

try:
    raise MyError("This is my custom exception.")

except MyError as e:
    print("Caught:", e)