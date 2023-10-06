import numpy as np

# 1. Create a 1D array of numbers from 0 to 9
a = np.arange(10)

print(a)

# 2. Create a 3×3 numpy array of all True’s

c = np.ones((3, 3), dtype=bool)

print(c)

my_array = np.array([[1, 2, 3], [4, 5, 6]])

print(my_array)

# Array Manipulation

reshaped_array = my_array.reshape(3, 2)

print(reshaped_array)

# universal functions

print(np.sum(my_array))

square_root = np.sqrt(my_array)

print(square_root)

squared_array = np.square(my_array)

print(squared_array)

# Array Broadcasting 

array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([1, 2, 3])

print(array1 + array2)

# Array Iteration

for x in array1:
    print(x)

# Array Joining

array3 = np.array([[7, 8, 9], [10, 11, 12]])

print(np.concatenate((array1, array3), axis=1))

# Array Splitting

print(np.array_split(array1, 2))

