def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner

#makepretty(ordinary)
@make_pretty
def ordinary():
    print("I am ordinary")

ordinary()