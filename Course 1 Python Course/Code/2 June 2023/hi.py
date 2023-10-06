name = "sid"

print("Hello, " + name + "!")

age = 20

print("You are " + str(age) + " years old!")

height = 1.75

print("You are " + str(height) + " meters tall!")

is_student = True

print("Are you a student? " + str(is_student))

result = 1 + 2

print("1 + 2 = " + str(result))

result = 10 / 3

print("10 / 3 = " + str(result))

result = 10 // 3

print("10 // 3 = " + str(result))

result = 10 % 3

print("10 % 3 = " + str(result))

is_equal = 1 == 1

print("1 == 1: " + str(is_equal))

is_greater = 1 > 2

print("1 > 2: " + str(is_greater))

is_less = 1 < 2

print("1 < 2: " + str(is_less))

is_greater_equal = 1 >= 2

print("1 >= 2: " + str(is_greater_equal))

if age >= 18:
    print("You are an adult!")
else:
    print("You are a child!")

for i in range(5):
    print(i)

i= 0

while i < 5:
    print(i)
    i += 1

def add(a, b):
    return a + b

result = add(1, 2)

print("1 + 2 = " + str(result))

import math

result = math.sqrt(9)

print("sqrt(9) = " + str(result))