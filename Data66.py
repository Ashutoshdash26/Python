# import csv
# with open ("study.csv","r")as file:
#     reader=csv.reader(file)
#     for row in reader:
#         print(row)

# def adder(*num):
#     sum=0
#     print(*num)
#     for n in num:
#         sum=sum+n
#     print("Sum : ",sum)
# adder()
# adder(10,20,30,40)


# def adder(*num):
#     sum=0
#     print(*num)
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



# g=lambda x,y,z:x**y+z
# print(g(5,6,7))
# check_age=lambda age:"Adult" if age>=18 else "Minor"
# print(check_age(25))

# l1=[4,5,6,7,8,9]
# print(list(map(lambda x:x*x,l1)))
# print(list(filter(lambda z:z%2==0,l1)))

# from functools import reduce
# print(reduce(lambda x,y:x+y,l1))

# # --- Part 2: enumerate ---
# languages = ['Python', 'Java', 'JavaScript']
# enumerate_prime = enumerate(languages)

# # Convert enumerate object to list
# print(list(enumerate_prime))



# x = -200
# print(abs(x))


def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
        print("1")
    return inner

#makepretty(ordinary)
@make_pretty
def ordinary():
    print("I am ordinary")

ordinary()