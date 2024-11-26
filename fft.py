import numpy as np
import matplotlib as plt


import numpy as np

# Create a numpy array
array = np.array([1, 2, 3, 4, 5])

# Calculate the mean of the array
mean_value = np.mean(array)
print("Mean of the array:", mean_value)

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plotting
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()