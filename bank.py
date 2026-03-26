class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print("Deposited:", amount)

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print("Withdrawn:", amount)
        else:
            print("Insufficient balance")

    def show_balance(self):
        print("Balance:", self.balance)




acc = BankAccount(1000)
acc.deposit(500)
acc.withdraw(300)
acc.show_balance()




class Person:
    def __init__(self, name):
        self.name = name

    def show(self):
        print("Name:", self.name)


class Employee(Person):
    def __init__(self, name, salary):
        super().__init__(name)
        self.salary = salary

    def show_salary(self):
        print("Salary:", self.salary)


e1 = Employee("Ashutosh", 50000)
e1.show()
e1.show_salary()