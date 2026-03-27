def bark(name="roger"):
    print(f"Woof woof ... {name}")


class animal:
    def walk(self):
        print("Walking ... ")

class Parrot(animal):
    def __init__(self,name):
        self.name=name
    def display(self):
        print(f"The parrot name is {self.name}")

class Book:
    def __init__(self, title):
        self.title = title
        self.available = True

    def borrow(self):
        if self.available:
            self.available = False
            print(f"{self.title} borrowed")
        else:
            print("Not available")

    def return_book(self):
        self.available = True
        print(f"{self.title} returned")


# b1 = Book("Python Basics")
# b1.borrow()
# b1.borrow()
# b1.return_book()