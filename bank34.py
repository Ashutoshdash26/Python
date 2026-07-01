class InsufficientBalance(Exception):
    pass

balance = 5000
amount = int(input("Enter withdrawal amount: "))

try:
    if amount > balance:
        raise InsufficientBalance("Insufficient Balance")

    balance -= amount
    print("Remaining Balance:", balance)

except InsufficientBalance as e:
    print(e)