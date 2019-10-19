import matplotlib.pyplot as plt
import numpy as np

matrix = np.array([[87,13], [33,46]])

'plot confusion matrix'
fig,ax = plt.subplots(figsize = (4, 4))
ax.matshow(matrix, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(x = j, y = i, s = matrix[i,j], va = "center", ha = "center")


plt.xlabel("predicted label")
plt.ylabel("true label")
plt.show()
