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


f=lambda x:x*x
print(f(8))

g=lambda x,y,z:x**y+z
print(g(5,6,7))
check_age=lambda age:"Adult" if age>=18 else "Minor"
print(check_age(25))


#map filter reduce

l1=[3,4,5,7.3,9]
print(list(map(lambda x:x*x,l1)))
print(tuple(map(lambda x:x*x,l1)))
print(set(map(lambda x:x*x,l1)))