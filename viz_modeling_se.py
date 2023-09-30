import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib import cm  
from mpl_toolkits.mplot3d import Axes3D  
import random  
# dummy random data  
x = np.arange(0, 101, 1)  
y = np.arange(0, 101, 1)  
peaks = 10
xc = np.max(x) * np.random.rand(peaks, 1)  # x center  
yc = np.max(y) * np.random.rand(peaks, 1)  # y center  
sigmax = 2 + 7 * np.random.rand(peaks, 1)  # x sigma  
sigmay = 3 + 5 * np.random.rand(peaks, 1)  # y sigma  
A = np.random.rand(peaks,1) -0.5 # peak amplitude  
  
nb_peaks = len(xc)  
  
xx, yy = np.meshgrid(x, y)  
  
z = 0  
for ck in range(nb_peaks):  
    z = z +A[ck] * np.exp(-(xx - xc[ck])**2 / (2 * sigmax[ck]**2) - (yy - yc[ck])**2 / (2 * sigmay[ck]**2))  
  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  

# Make the colormap as grid
my_colormap = cm.jet(np.arange(cm.jet.N)) 

my_colormap[:, -1] = 0.1  # Set the alpha value to 0.5 for all colors  
my_colormap = cm.colors.ListedColormap(my_colormap)  
  
surf = ax.plot_surface(xx, yy, z, cmap=my_colormap, linewidth=1, antialiased=False, alpha=0.5)  
contours = ax.contour(xx, yy, z, levels=10, offset=np.min(z), cmap='viridis', zdir='z', alpha=0.5)
#fig.colorbar(surf, shrink=0.5, aspect=5) 
plt.axis('off')
plt.title('Random Function')    

plt.show()  



# write a function that generate the upset plot with five lists as the input
# the five lists are:
# 1. the list of all cell types
# 2. the list of cell types in the first group
# 3. the list of cell types in the second group
# 4. the list of cell types in the third group
# 5. the list of cell types in the fourth group
# 6. the list of cell types in the fifth group

# the output is the upset plot
# the output is the upset plot

# make a data frame for each set
set1_df = pd.DataFrame({'set1': True, 'Name': set1})
set2_df = pd.DataFrame({'set2': True, 'Name': set2})
set3_df = pd.DataFrame({'set3': True, 'Name': set3})
set4_df = pd.DataFrame({'set4': True, 'Name': set4})
set5_df = pd.DataFrame({'set5': True, 'Name': set5})

# Merge three data frames together
animals_df = set1_df.merge(
    set2_df.merge(set3_df.merge(set4_df.merge(set5_df, how='outer'), how='outer'), how='outer'), how='outer')