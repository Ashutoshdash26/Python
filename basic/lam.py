lambda num:num*2
multiplication= lambda a,b:a*b
print(multiplication(2,4))
#map , filter , reduce
def double(a):
    return a*2
b=double(5)
print(b)

num=[1,2,3]
result=map(double,num)
print(list(result))

#MAP
nums = [1, 2, 3, 4]

result = list(map(lambda x: x * 2, nums))
print(result)

#FILTER
nums = [1, 2, 3, 4, 5]

even = list(filter(lambda x: x % 2 == 0, nums))
print(even)

#SORTED

students = [("Ashu", 22), ("Ravi", 20), ("Neha", 21)]

sorted_list = sorted(students, key=lambda x: x[1])
print(sorted_list)

check = lambda x: "Even" if x % 2 == 0 else "Odd"
print(check(4))