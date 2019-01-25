import matplotlib.pyplot as plt
import numpy as np

glod, chihh = 400, 400

glod_height = 40 + 10 * np.random.randn(glod)
chihh_height = 26 + 6 * np.random.randn(chihh)

plt.hist([glod_height, chihh_height], stacked=True, color=['r', 'b'])
plt.show()
