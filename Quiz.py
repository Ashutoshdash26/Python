score = 0

print("🧠 Quiz Game")

q1 = input("Capital of India? ")
if q1.lower() == "delhi":
    score += 1

q2 = input("5 + 3 = ? ")
if q2 == "8":
    score += 1

q3 = input("Python is a (language/animal)? ")
if q3.lower() == "language":
    score += 1

print("Your score:", score)