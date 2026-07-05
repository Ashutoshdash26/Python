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

def perform(**kwargs):
    print(kwargs)
    print(type(kwargs))

perform(banana=5, mango=10, cherry=4)