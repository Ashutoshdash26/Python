import re

# ==========================================
# 1. Largest Prime Less Than n
# ==========================================
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def largest_prime_less_than(n):
    for i in range(n - 1, 1, -1):
        if is_prime(i):
            return i
    return None

def run_task_1():
    print("\n--- Task 1: Largest Prime Less Than n ---")
    try:
        n = int(input("Enter a value for n: "))
        result = largest_prime_less_than(n)
        if result:
            print(f"The largest prime number less than {n} is: {result}")
        else:
            print(f"There is no prime number less than {n}.")
    except ValueError:
        print("Invalid input. Please enter an integer.")


# ==========================================
# 2. Intersection of Two Lists
# ==========================================
def findcommon(list1, list2):
    set2 = set(list2)
    common = []
    for item in list1:
        if item in set2 and item not in common:
            common.append(item)
    return common


# ==========================================
# 3. Leap Year Checker
# ==========================================
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


# ==========================================
# 4. Sum of Positive Numbers
# ==========================================
def positivesum(lst):
    return sum(x for x in lst if x > 0)


# ==========================================
# 5. Square of Each Element
# ==========================================
def listsquare(lst):
    return [x**2 for x in lst]


# ==========================================
# 6. Area of a Rectangle Function
# ==========================================
def cal_area(length, width):
    return length * width


# ==========================================
# 7. Area of a Rectangle using Lambda
# ==========================================
calc_area_lambda = lambda length, width: length * width


# ==========================================
# 8. Number Operator Function
# ==========================================
def operate_numbers(num1, num2, op):
    if op == '+': return num1 + num2
    elif op == '-': return num1 - num2
    elif op == '*': return num1 * num2
    elif op == '/': return num1 / num2 if num2 != 0 else "Error: Division by zero"
    elif op == '%': return num1 % num2 if num2 != 0 else "Error: Division by zero"
    elif op == '//': return num1 // num2 if num2 != 0 else "Error: Division by zero"
    elif op == '**': return num1 ** num2
    else: return "Invalid Operator"


# ==========================================
# 9. Tax Bracket Calculator
# ==========================================
def calculate_tax(salary):
    # Salary inputs assumed in Lakhs (e.g., 10 for 10 Lakh)
    if salary <= 3:
        return 0
    elif salary <= 6:
        return (salary - 3) * 0.05
    elif salary <= 9:
        return (3 * 0.05) + (salary - 6) * 0.10
    elif salary <= 12:
        return (3 * 0.05) + (3 * 0.10) + (salary - 9) * 0.15
    elif salary <= 15:
        return (3 * 0.05) + (3 * 0.10) + (3 * 0.15) + (salary - 12) * 0.20
    else:
        return (3 * 0.05) + (3 * 0.10) + (3 * 0.15) + (3 * 0.20) + (salary - 15) * 0.30


# ==========================================
# 10. Distinct Sorted Word Sequence
# ==========================================
def run_task_10():
    print("\n--- Task 10: Distinct Sorted Words ---")
    user_input = input("Enter comma-separated words: ")
    words = [word.strip() for word in user_input.split(',')]
    distinct_sorted_words = sorted(list(set(words)))
    print("Result:", ", ".join(distinct_sorted_words))


# ==========================================
# 11. File Analyzer
# ==========================================
def analyze_file(filename):
    lines = 0
    words = 0
    characters = 0
    char_a_count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                lines += 1
                characters += len(line)
                char_a_count += line.lower().count('a')
                words += len(line.split())
        print(f"Lines: {lines}\nWords: {words}\nTotal Characters: {characters}\nOccurrences of 'a': {char_a_count}")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


# ==========================================
# 12. Copy File Excluding "the"
# ==========================================
def copy_excluding_the(source_file, dest_file):
    try:
        with open(source_file, 'r') as src, open(dest_file, 'w') as dest:
            for line in src:
                words = line.split()
                filtered_words = [word for word in words if word.lower() != "the"]
                dest.write(" ".join(filtered_words) + "\n")
        print(f"Content copied safely to {dest_file} without the word 'the'.")
    except FileNotFoundError:
        print(f"Source file '{source_file}' not found.")


# ==========================================
# 13. Search for a String in a File
# ==========================================
def search_string_in_file(filename, search_str):
    found = False
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if search_str in line:
                    print(f"Found '{search_str}' on line {line_num}: {line.strip()}")
                    found = True
        if not found:
            print(f"'{search_str}' not found in the file.")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


# ==========================================
# 14. Sort Words ignoring Special Characters
# ==========================================
def sort_words_clean(text):
    words = re.findall(r'\b\w+\b', text)
    return sorted(words, key=str.lower)


# ==========================================
# 15. Force Valid Integer Input
# ==========================================
def get_valid_integer():
    user_input = input("Please enter an integer: ")
    if not user_input.lstrip('-').isdigit():
        raise ValueError("ValueError: The input is not a valid integer.")
    return int(user_input)

def run_task_15():
    print("\n--- Task 15: Valid Integer Verification ---")
    try:
        val = get_valid_integer()
        print(f"Success! Your integer is: {val}")
    except ValueError as e:
        print(e)


# ==========================================
# Execution / Test Cases Demo
# ==========================================
if __name__ == "__main__":
    print("--- Demonstrating Static Functions ---")
    
    # Task 2
    print("Task 2 (Common):", findcommon([1, 2, 3, 4], [3, 4, 5, 6]))
    
    # Task 3
    print("Task 3 (Leap Year 2024):", is_leap_year(2024))
    
    # Task 4
    print("Task 4 (Positive Sum):", positivesum([1, 500, -5, 6, -7, 9, -100]))
    
    # Task 5
    print("Task 5 (List Square):", listsquare([1, 3, 4, 5, 10]))
    
    # Task 6 & 7
    print("Task 6 (Area Function):", cal_area(10, 5))
    print("Task 7 (Area Lambda):", calc_area_lambda(10, 5))
    
    # Task 8
    print("Task 8 (Operate +):", operate_numbers(5, 3, '+'))
    print("Task 8 (Operate -):", operate_numbers(10, 2, '-'))
    
    # Task 9
    print("Task 9 (Tax for 10 Lakh):", calculate_tax(10))
    
    # Task 14
    print("Task 14 (Clean Sort):", sort_words_clean("Hello world! This is a test, clean-cut puzzle."))

    # Interactive test calls (Uncomment any to run)
    # run_task_1()
    # run_task_10()
    # run_task_15()