# if(True & False):
#     print("Ashutosh")
# print("Dash")
# name=input("What is your namr ? ")
# print(name + " is my name")
# print(abs(-5.7))
# print(round(5.54))
from enum import Enum
class State(Enum):
    INCATIVE=0
    ACTIVE=1
print(State["ACTIVE"].value)
print(State.ACTIVE)         # State.ACTIVE
print(State.ACTIVE.value)   # 1
print(State["ACTIVE"])      # State.ACTIVE
print(State["ACTIVE"].value)# 1