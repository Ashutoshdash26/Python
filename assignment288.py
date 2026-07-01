from datetime import date

# =====================================================================
# Q1: Write a Python program that accepts your date of birth as a 
# parameter and prints your age.
# =====================================================================
def calculate_age(birth_date):
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

print("--- Q1: Age Calculator ---")
my_dob = date(2000, 5, 15)  # Replace with actual birth date
print(f"Age for DOB {my_dob}: {calculate_age(my_dob)}\n")


# =====================================================================
# Q2: Write a program using the for loop to print all the prime numbers 
# less than 100. Rewrite the same program using a while loop.
# =====================================================================
print("--- Q2: Prime Numbers less than 100 ---")
print("Using For Loop:")
for num in range(2, 100):
    is_prime = True
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        print(num, end=" ")
print("\n")

print("Using While Loop:")
num = 2
while num < 100:
    is_prime = True
    i = 2
    while i * i <= num:
        if num % i == 0:
            is_prime = False
            break
        i += 1
    if is_prime:
        print(num, end=" ")
    num += 1
print("\n\n")


# =====================================================================
# Q3: Create a program that takes a sentence as input and counts the 
# number of words in it.
# =====================================================================
print("--- Q3: Word Count ---")
sample_sentence = "Python programming is fun and efficient"
words_count = len(sample_sentence.split())
print(f"Sentence: '{sample_sentence}'")
print(f"Word Count: {words_count}\n")


# =====================================================================
# Q4: Create a program that takes a temperature in Celsius and converts 
# it into Fahrenheit.
# =====================================================================
print("--- Q4: Celsius to Fahrenheit ---")
celsius_val = 37.5
fahrenheit_val = (celsius_val * 9/5) + 32
print(f"{celsius_val}°C is equal to {fahrenheit_val}°F\n")


# =====================================================================
# Q5: Implement a program that converts a given number of minutes 
# into hours and minutes.
# =====================================================================
print("--- Q5: Minutes to Hours & Minutes ---")
total_minutes = 135
hours = total_minutes // 60
remaining_mins = total_minutes % 60
print(f"{total_minutes} minutes = {hours} hours and {remaining_mins} minutes\n")


# =====================================================================
# Q6: Given the following list with values and keys, write a program to 
# convert it into the form of a list of dictionaries with key-value pairs, 
# sort on id, and print.
# Input_list = ["morning", 1, "evening", 3, "afternoon", 2]
# =====================================================================
print("--- Q6: Convert List to Sorted List of Dictionaries ---")
Input_list = ["morning", 1, "evening", 3, "afternoon", 2]
dict_list = []
for i in range(0, len(Input_list), 2):
    dict_list.append({"value": Input_list[i], "id": Input_list[i+1]})
dict_list.sort(key=lambda x: x["id"])
print(f"Sorted Dictionaries: {dict_list}\n")


# =====================================================================
# Q7: Write a program that finds the common elements between two lists 
# and stores them in a new list.
# Input : [1,2,3,4,5] , [1,5,6,7,8]
# Output : [1,5]
# =====================================================================
print("--- Q7: Common Elements between Lists ---")
list1 = [1, 2, 3, 4, 5]
list2 = [1, 5, 6, 7, 8]
common_elements = [item for item in list1 if item in list2]
print(f"Common List: {common_elements}\n")


# =====================================================================
# Q8: Write a program that checks if a given list is sorted in 
# ascending order or not. Input : [-10,2,5,6,8,90]
# =====================================================================
print("--- Q8: Check if List is Sorted ---")
sorted_check_list = [-10, 2, 5, 6, 8, 90]
if sorted_check_list == sorted(sorted_check_list):
    print("Yes List is Sorted in Ascending Order\n")
else:
    print("No List is Not Sorted in Ascending Order\n")


# =====================================================================
# Q9: Write a Python program to count the occurrences of each element 
# in each list. Input [1,3,4,1,1,1,3]
# =====================================================================
print("--- Q9: Element Frequencies ---")
freq_list = [1, 3, 4, 1, 1, 1, 3]
frequencies = {}
for item in freq_list:
    frequencies[item] = frequencies.get(item, 0) + 1
for num, count in frequencies.items():
    print(f"Count of {num} is {count}")
print()


# =====================================================================
# Q10: Write a program to find the number of occurrences of each 
# character present in the given String. Input: ABCABCABBDE
# =====================================================================
print("--- Q10: Character Frequencies ---")
input_string = "ABCABCABBDE"
char_counts = {}
for char in input_string:
    char_counts[char] = char_counts.get(char, 0) + 1
output_format = ",".join([f"{k}-{v}" for k, v in char_counts.items()])
print(f"Output: {output_format}\n")


# =====================================================================
# Q11: Find out if entered number is Armstrong or not
# =====================================================================
print("--- Q11: Armstrong Number Check ---")
test_num = 153
num_str = str(test_num)
power = len(num_str)
is_armstrong = test_num == sum(int(digit) ** power for digit in num_str)
print(f"Is {test_num} an Armstrong number?: {is_armstrong}\n")


# =====================================================================
# Q12: Print index of vowels from a string
# =====================================================================
print("--- Q12: Vowel Indices ---")
vowel_str = "Hello Deloitte"
vowels = "aeiouAEIOU"
indices = [idx for idx, char in enumerate(vowel_str) if char in vowels]
print(f"Indices of vowels in '{vowel_str}': {indices}\n")


# =====================================================================
# Q13: Calculate a sum of squares of even digits from a user entered number
# =====================================================================
print("--- Q13: Sum of Squares of Even Digits ---")
digit_str = "2456"
sum_even_squares = sum(int(d)**2 for d in digit_str if int(d) % 2 == 0)
print(f"Sum of squares of even digits in '{digit_str}': {sum_even_squares}\n")


# =====================================================================
# Q14: Accept percentage value from user and print grade as per below conditions:
# per > 90 --> outstanding | 90 > per > 75 --> first class | 75 > per > 60 --> second class | per < 60 --> average
# =====================================================================
print("--- Q14: Percentage Grade Allocator ---")
def check_grade(per):
    if per > 90:
        return "outstanding"
    elif per > 75:
        return "first class"
    elif per > 60:
        return "second class"
    else:
        return "average"

print(f"Grade for 85%: {check_grade(85)}\n")


# =====================================================================
# Q15: Using keyword Argument, calculate the updated salary as per conditions.
# =====================================================================
print("--- Q15: Salary Hike Calculation ---")
def calculateSalary(name, sal, dept):
    if dept == 'IT':
        hike = 0.15
    elif dept == 'Audit':
        hike = 0.10
    elif dept == 'Tax':
        hike = 0.12
    else:
        hike = 0.05
    updated_sal = sal + (sal * hike)
    print(f"The updated salary of {name} is {int(updated_sal)}")

calculateSalary('AMAR', sal=10000, dept='IT')
calculateSalary('SANJAY', sal=20000, dept='Audit')
calculateSalary('Aditya', sal=15000, dept='other')
print()


# =====================================================================
# Q16: Write a Python program that accepts a comma-separated sequence 
# of words as input and prints the distinct words in sorted form.
# =====================================================================
print("--- Q16: Distinct Sorted Words ---")
sample_words = "red, white, black, red, green, black"
distinct_words = sorted(list(set([w.strip() for w in sample_words.split(",")])))
print(f"Expected Result: {', '.join(distinct_words)}\n")


# =====================================================================
# Q17: Write a Python program to create a Substitution Cipher encryption.
# =====================================================================
print("--- Q17: Substitution Cipher (+1 Shift) ---")
def substitution_cipher(text):
    result = []
    for char in text:
        if char.isalpha():
            if char == 'z': result.append('a')
            elif char == 'Z': result.append('A')
            else: result.append(chr(ord(char) + 1))
        else:
            result.append(char)
    return "".join(result)

cipher_text = "I am working in deloitte from 5 years."
print(f"Encrypted sentence: {substitution_cipher(cipher_text)}\n")


# =====================================================================
# Q18: Write a Python function that uses a lambda function to find the 
# maximum value in a list of tuples based on the second element.
# =====================================================================
print("--- Q18: Find Max in Tuple List via Lambda ---")
tuple_list = [(10, 50), (30, 100), (40, 20), (5, 80)]
max_val = max(tuple_list, key=lambda x: x[1])
print(f"Maximum tuple based on 2nd element: {max_val}\n")


# =====================================================================
# Q19: Nested List Retrieval Operations
# =====================================================================
print("--- Q19: Nested List Data Extraction ---")
l1 = [100, 1000, [10, 20, 30, [50, 60], [70, 80], 15, 25], (90, 200, 500), 2000]

print("retrieve 20:      ", l1[2][1])
print("retrieve [50, 60]:", l1[2][3])
print("retrieve 25:      ", l1[2][6])
print("retrieve 200:     ", l1[3][1])
print("retrieve 80:      ", l1[2][4][1])
print()


# =====================================================================
# Q20: Nested Dictionary and List Data Extraction
# =====================================================================
print("--- Q20: Nested Dictionary Data Extraction ---")
d1 = {
    'a': [1, 2, 3], 
    'b': {1: 'One', 10: 'Ten'}, 
    'c': [10, 20, 30, 40, 50], 
    'd': {'e': {100: ('a', 'c', 'f')}}
}

print("(a) retrieve 'a':    ", d1['d']['e'][100][0])
print("(b) retrieve [20, 30]:", d1['c'][1:3])
print("(c) retrieve 'Ten':  ", d1['b'][10])
print("(d) retrieve 3:      ", d1['a'][2])