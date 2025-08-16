print("Ashutosh Dash")
import pyjokes
joke=pyjokes.get_joke()
print("Printing jokes ... ")
print(joke)
import pyttsx3


engine = pyttsx3.init()

engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Convert text to speech
engine.say("Hi my name is Ashutosh Dash ")
engine.runAndWait()


import os

# Specify the directory (you can change this to any path)
directory =  "/Users\KIIT\Music\list"

# List all files and folders in the directory
print(f"Contents of directory: {os.path.abspath(directory)}\n")
for item in os.listdir(directory):
    print(item)
