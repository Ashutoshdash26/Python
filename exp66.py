# if(True & False):
#     print("Ashutosh")
# print("Dash")
# name=input("What is your namr ? ")
# print(name + " is my name")
# print(abs(-5.7))
# print(round(5.54))

###enum
# from enum import Enum
# class State(Enum):
#     INCATIVE=0
#     ACTIVE=1
# print(State["ACTIVE"].value)
# print(State.ACTIVE)         # State.ACTIVE
# print(State.ACTIVE.value)   # 1
# print(State["ACTIVE"])      # State.ACTIVE
# print(State["ACTIVE"].value)# 1

# print(list(State))


### User input 


# age=int(input("Enter your age : "))
# print(f"Your age is {age}")
# print(type(age))

## function 


def hello(name="Ashutosh ", age ="23"):
    print(f"my name is {name} and age is {age}")

hello()
hello("Parth")
hello("parth",23)