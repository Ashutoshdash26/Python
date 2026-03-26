# Parent Class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("Animal makes a sound")

    def walk(self):
        print(f"{self.name} is walking")


# Child Class (Inheritance)
class Dog(Animal):
    def __init__(self, name, age):
        super().__init__(name)   # calling parent constructor
        self.age = age

    # Method Overriding (Polymorphism)
    def speak(self):
        print(f"{self.name} says Woof!")

    def bark(self):
        return "Woof Woof!"

    # Encapsulation (getter & setter)
    def set_age(self, age):
        self.age = age

    def get_age(self):
        return self.age

    # String representation
    def __str__(self):
        return f"Dog(Name: {self.name}, Age: {self.age})"


# Another Child Class (Polymorphism)
class Cat(Animal):
    def speak(self):
        print(f"{self.name} says Meow!")


# Function demonstrating Polymorphism
def animal_sound(animal):
    animal.speak()


# Main Program
dog1 = Dog("Buddy", 3)
cat1 = Cat("Kitty")

print(dog1)              # __str__
dog1.walk()              # inherited method
dog1.speak()             # overridden method
print(dog1.bark())       # own method

# Encapsulation
dog1.set_age(5)
print("Updated Age:", dog1.get_age())

# Polymorphism
animal_sound(dog1)
animal_sound(cat1)

def hell():
    print("Hello Ashutosh Das")