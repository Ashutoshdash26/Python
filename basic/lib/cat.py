class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("Animal makes a sound")

    def walk(self):
        print(f"{self.name} is walking")


class Dog(Animal):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

    def speak(self):
        print(f"{self.name} says Woof!")

    def bark(self):
        return "Woof Woof!"

    def set_age(self, age):
        self.age = age

    def get_age(self):
        return self.age

    def __str__(self):
        return f"Dog(Name: {self.name}, Age: {self.age})"


class Cat(Animal):
    def speak(self):
        print(f"{self.name} says Meow!")


def animal_sound(animal):
    animal.speak()


def hell():
    print("Hello Ashutosh Das")


# ✅ ONLY ONE main block
if __name__ == "__main__":
    dog1 = Dog("Buddy", 3)
    cat1 = Cat("Kitty")

    print(dog1)
    dog1.walk()
    dog1.speak()
    print(dog1.bark())

    dog1.set_age(5)
    print("Updated Age:", dog1.get_age())

    animal_sound(dog1)
    animal_sound(cat1)