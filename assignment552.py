# ==============================================================================
# QUESTION 1: Swap two numbers without using a temporary variable.
# ==============================================================================
print("--- Question 1 ---")
a = 10
b = 20
print(f"Before swapping: a = {a}, b = {b}")

# Swapping using Python's tuple assignment
a, b = b, a
print(f"After swapping: a = {a}, b = {b}\n")


# ==============================================================================
# QUESTION 2: Explain the difference between `is` and `==` with an example.
# ==============================================================================
print("--- Question 2 ---")
# == checks for equality of values; is checks for identity (same memory address)
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1

print("list1 == list2:", list1 == list2)  # True because values are identical
print("list1 is list2:", list1 is list2)  # False because they are distinct memory objects
print("list1 is list3:", list1 is list3)  # True because they reference the same object
print()


# ==============================================================================
# QUESTION 3: Convert a binary number (given as a string) to its decimal equivalent.
# ==============================================================================
print("--- Question 3 ---")
binary_str = "1101"
decimal_val = int(binary_str, 2)
print(f"Binary string '{binary_str}' in decimal is: {decimal_val}\n")


# ==============================================================================
# QUESTION 4: Find the Fibonacci series of a given number of terms.
# ==============================================================================
print("--- Question 4 ---")
n = 7
x, y = 0, 1
fib_series = []
for _ in range(n):
    fib_series.append(x)
    x, y = y, x + y
print(f"Fibonacci series up to {n} terms: {fib_series}\n")


# ==============================================================================
# QUESTION 5: Perform bitwise operations on two integers and explain results.
# ==============================================================================
print("--- Question 5 ---")
val1 = 6  # Binary: 0110
val2 = 3  # Binary: 0011

print(f"Bitwise AND (val1 & val2): {val1 & val2}")    # 0110 & 0011 = 0010 (2)
print(f"Bitwise OR (val1 | val2): {val1 | val2}")     # 0110 | 0011 = 0111 (7)
print(f"Bitwise XOR (val1 ^ val2): {val1 ^ val2}")    # 0110 ^ 0011 = 0101 (5)
print(f"Bitwise NOT (~val1): {~val1}")                # ~6 = -(6+1) = -7
print()


# ==============================================================================
# QUESTION 6: Check whether a string is a palindrome without string slicing.
# ==============================================================================
print("--- Question 6 ---")
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

word = "radar"
print(f"Is '{word}' a palindrome?: {is_palindrome(word)}\n")


# ==============================================================================
# QUESTION 7: Demonstrate ternary and membership operators.
# ==============================================================================
print("--- Question 7 ---")
# Ternary Operator
age = 20
status = "Adult" if age >= 18 else "Minor"
print(f"Ternary Result: {status}")

# Membership Operator
fruits = ["apple", "banana", "cherry"]
print(f"Membership Result ('banana' in fruits): {'banana' in fruits}\n")


# ==============================================================================
# QUESTION 8: Count the total number of digits in a number.
# ==============================================================================
print("--- Question 8 ---")
num = 987654
digit_count = len(str(abs(num)))
print(f"The number of digits in {num} is: {digit_count}\n")


# ==============================================================================
# QUESTION 9: Perform membership and identity operations on two integers.
# ==============================================================================
print("--- Question 9 ---")
num1 = 500
num2 = 500
check_list = [100, 200, 300, 500]

print(f"Is num1 in check_list? (Membership): {num1 in check_list}") 
print(f"Is num1 identity equal to num2? (num1 is num2): {num1 is num2}\n")


# ==============================================================================
# QUESTION 10: Unpack a list or array elements.
# ==============================================================================
print("--- Question 10 ---")
colors = ["red", "green", "blue"]
c1, c2, c3 = colors
print(f"Unpacked variables -> c1: {c1}, c2: {c2}, c3: {c3}\n")


# ==============================================================================
# QUESTION 11: Count the occurrences of digits in a given number.
# ==============================================================================
print("--- Question 11 ---")
num_str = "9032198232"
digit_input = "2"
occurrences = num_str.count(digit_input)
print(f"The digit {digit_input} has occurred - {occurrences} times.\n")


# ==============================================================================
# QUESTION 12: Unpack a tuple into variables.
# ==============================================================================
print("--- Question 12 ---")
point = (10, 20, 30)
x_coord, y_coord, z_coord = point
print(f"Unpacked tuple -> x: {x_coord}, y: {y_coord}, z: {z_coord}\n")


# ==============================================================================
# QUESTION 13: Dictionary manipulations on student data.
# ==============================================================================
print("--- Question 13 ---")
student = {'name': 'John', 'age': 21, 'courses': ['Math', 'Science']}

# Retrieve name
print("Retrieve 'name':", student['name'])

# Update age
student['age'] = 22

# Add grade
student['grade'] = 'A'

# Add Hindi to courses
student['courses'].append('Hindi')

# Get all keys
print("Keys:", list(student.keys()))

# Get all values
print("Values:", list(student.values()))

# Remove courses
student.pop('courses')
print("Final Student Dictionary:", student)
print()


# ==============================================================================
# QUESTION 14: Nested Dictionary Activity - Phone Book Operations.
# ==============================================================================
print("--- Question 14 ---")
phone_book = {
    'Alice': {'phone': '9876543210', 'city': 'Bangalore', 'email': 'alice@gmail.com'},
    'Bob': {'phone': '8765432109', 'city': 'Mumbai', 'email': 'bob@yahoo.com'},
    'John': {'phone': '7654321098', 'city': 'Bangalore', 'email': 'john@gmail.com'},
    'Goutam': {'phone': '6543210987', 'city': 'Bhuvaneshwar', 'email': 'goutam@outlook.com'}
}

# a. Retrieve phone number
print("a. Alice's Phone:", phone_book['Alice']['phone'])

# b. Update email
phone_book['Bob']['email'] = 'bob_new@yahoo.com'

# c. Add a new person
phone_book['Charlie'] = {'phone': '5432109876', 'city': 'Delhi', 'email': 'charlie@gmail.com'}

# d. Retrieve people in Bangalore
bangalore_residents = [name for name, details in phone_book.items() if details['city'] == 'Bangalore']
print("d. People in Bangalore:", bangalore_residents)

# e. Retrieve people with gmail.com
gmail_users = [name for name, details in phone_book.items() if details['email'].endswith('gmail.com')]
print("e. People with Gmail:", gmail_users)

# f. Retrieve people whose name contains any vowel
vowels = set("aeiouAEIOU")
vowel_names = [name for name in phone_book.keys() if any(char in vowels for char in name)]
print("f. People with vowels in name:", vowel_names)
print()


# ==============================================================================
# QUESTION 15: Print dictionary data in specific text string format.
# ==============================================================================
print("--- Question 15 ---")
nested_dict = {
    'Goutam': {'city': 'Bhuvaneshwar', 'age': 23}
}

for name, details in nested_dict.items():
    print(f' "{name} resides in {details["city"]} city and his age is {details["age"]} "\n')