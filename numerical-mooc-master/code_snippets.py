print("Hello World")

# creating simple numpy arrays and operations
import numpy as np

a = np.linspace(1,10,1)
print(a)

y = a**3
print(y)

# plotting simple plots using matplotlib

import matplotlib.pyplot as plt

plt.plot(a,y) #plots the graph
plt.show() #shows the plot
