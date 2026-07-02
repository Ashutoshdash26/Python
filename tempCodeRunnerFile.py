def myfun(arg1, *argv):
    print("First Argument:", arg1)

    for arg in argv:
        print("Argument:", arg)

    print("#" * 20)
    print("The argv:", argv)
    print("#" * 20)