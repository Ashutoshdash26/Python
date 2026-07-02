file = open(r"C:\Users\KIIT\OneDrive\Pictures\Screenshots\ab.png", "rb")

data = file.read()

print("Image read successfully!")
print("Image size:", len(data), "bytes")

file.close()

source = open(r"C:\Users\KIIT\OneDrive\Pictures\Screenshots\Screenshot 2026-06-22 111425.png", "rb")

destination = open("copy_ab.png", "wb")

destination.write(source.read())

source.close()
destination.close()

print("Image copied successfully!")





#----------------------------------------------------------------------------------------------


file = open("data.bin", "wb")

file.write(b"Hello Ashutosh")
file.write(b"\nWelcome to Python")

file.close()

print("Binary file created successfully!")




file = open("data.bin", "rb")

data = file.read()

print(data)

file.close()

file = open("data.bin", "rb")

data = file.read()

print(data.decode())

file.close()