
def say(name="Ashutosh Dash "):
    global count
    def talk():
        global count
        count=count+1
        print(f"Hi {name}, count is {count}")
    talk()

count=10
say()

    