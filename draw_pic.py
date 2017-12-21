import numpy as np
import matplotlib.pyplot as plt

history = open('history','r')
acc1 = eval(history.readline())

plt.plot(range(1,len(acc1)+1),acc1)
plt.show()