import fun
fun.hello()
from fun import hello
hello()
from lib.dog import hell
hell()

# from lib.dog import *
# dog1 = Dog("Buddy", 3)
# print(dog1.bark())


# from lib.cat import *

# dog1 = Dog("Buddy", 3)
# print(dog1.bark())


import lib.cat

dog1 = lib.cat.Dog("Buddy", 3)
print(dog1.bark())

cat1 = lib.cat.Cat("Kitty")
cat1.speak()

lib.cat.hell()