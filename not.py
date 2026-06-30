# 1a. Prime Numbers Less Than 100 (Using a For Loop)
print("--- 1a. Prime Numbers Less Than 100 (For Loop) ---")
for num in range(2, 100):
    is_prime = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        print(num, end=" ")
print("\n")


# 1b. Prime Numbers Less Than 100 (Using a While Loop)
print("--- 1b. Prime Numbers Less Than 100 (While Loop) ---")
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
print("\n")


# 2. Check if a Given Number is Positive, Negative, or Zero
print("--- 2. Positive, Negative, or Zero Checker ---")
check_num = float(input("Enter a number to check: "))
if check_num > 0:
    print("The number is positive.")
elif check_num < 0:
    print("The number is negative.")
else:
    print("The number is zero.")
print()


# 3. Count the Number of Words in a Sentence
print("--- 3. Word Counter ---")
sentence = input("Enter a sentence: ")
word_count = len(sentence.split())
print(f"The number of words in the sentence is: {word_count}\n")


# 4. Print the First 10 Even Numbers
print("--- 4. First 10 Even Numbers ---")
for i in range(1, 11):
    print(i * 2, end=" ")
print("\n")


# 5. Swap the Values of Two Variables
print("--- 5. Swap Two Variables ---")
var1 = input("Enter value for variable 1 (a): ")
var2 = input("Enter value for variable 2 (b): ")
var1, var2 = var2, var1
print(f"After swapping: a = {var1}, b = {var2}\n")


# 6. Convert Temperature from Celsius to Fahrenheit
print("--- 6. Celsius to Fahrenheit Converter ---")
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C is equal to {fahrenheit}°F\n")


# 7. Convert Minutes into Hours and Minutes
print("--- 7. Minutes to Hours & Minutes Converter ---")
total_minutes = int(input("Enter the total number of minutes: "))
hours = total_minutes // 60
remaining_minutes = total_minutes % 60
print(f"{total_minutes} minutes = {hours} hour(s) and {remaining_minutes} minute(s).\n")


# 8. Compare 3 Different Integers and Find the Biggest
print("--- 8. Find the Biggest of 3 Integers ---")
n1 = int(input("Enter first integer: "))
n2 = int(input("Enter second integer: "))
n3 = int(input("Enter third integer: "))

if n1 >= n2 and n1 >= n3:
    biggest = n1
elif n2 >= n1 and n2 >= n3:
    biggest = n2
else:
    biggest = n3

print(f"The biggest number among {n1}, {n2}, and {n3} is: {biggest}\n")