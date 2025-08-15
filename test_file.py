import numpy as np
# Test vector
test_vector = np.array(0) # This creates the initial vector
print(test_vector) # This prints the entire array
test_vector = np.append(test_vector, 2) # This appends a new value to the array
print(test_vector) # This prints the array
print(test_vector[1])

# Assume test_array is a 2D array of velocity vectors over time in the columns
# and the rows are the x and y components, respectively.

test_array = np.array([[0],[0]]) # This creates the initial 2D array
print(test_array) # This prints the entire array
test_array = np.append(test_array, [[2],[1]], axis=1) # This appends a new column to the array
print(test_array) # This prints the array
print(test_array[0]) # This gets all the x components of our pretend velocity vector
print(test_array[1]) # This gets the y components
print(test_array[:,1]) # This gets any single column (a single velocity vector at a time stamp)