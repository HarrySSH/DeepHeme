from math import sqrt, pi, sin, cos  
from PIL import Image, ImageFilter  
import glob
  
from math import sqrt, pi, sin, cos, degrees  
from PIL import Image, ImageFilter  
import cv2
import numpy as np
from math import sqrt, pi, sin, cos, degrees  
from PIL import Image, ImageFilter  
  
 

    
  
# List of image paths (replace with paths to your cell images)  

 
# List of image paths (replace with paths to your cell images)  
images = glob.glob('../../../../Documents/wbc_all_cells/*.jpg')

# sample 200 images,
images = images[:800]

# concatenate the images into a single images, 4 rows, 100 columns,
# each image is 96x96 pixels, 3 channels (RGB), 
# between each two images there is a image shape gap, 96*96 pixels

# 4 rows, 100 columns, 96*96 pixels, 3 channels (RGB),
# between each two images there is a image shape gap, 96*96 pixels
# 4 rows, 100 columns, 96*96 pixels, 3 channels (RGB),

# the canvas size is 96*100, 96*4 pixels, 3 channels (RGB),
# build the canvas with white color
canvas = np.ones((96*16+48*15, 96*100 , 3), dtype=np.uint8)*255

for j in range(0,16):
    for i in range(0,50):
    
        # read the image
        img = cv2.imread(images[i+j*50])
        # resize the image to 96*96 pixels
        #img = cv2.resize(img, (96, 96))
        # concatenate the image to the canvas
        if j % 2 == 0:
            canvas[j*96:(j+1)*96, i*2*96:(i*2+1)*96] = img
        else:
            canvas[j*96:(j+1)*96, (i*2+1)*96:(i*2+2)*96] = img
        
        

        
    if j == 15:
        break

    # add a image shape gap with 48 pixels between each two rows
    canvas[(j+1)*96+j*48:(j+1)*96+(j+1)*48, :] = 255




# flip the image vertically
canvas = cv2.flip(canvas, 0)
    
        
# show the canvas
cv2.imshow('canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the canvas
cv2.imwrite('canvas.png', canvas)

import cv2  
import numpy as np  
from tqdm import tqdm
  
import cv2  
import numpy as np  
  
def curve_image_circular(image_path, output_path):  

    # Read the image  
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    height, width = img.shape[:2]  
  
      
    max_radius = int(np.sqrt(width * width + height * height) / 2)  
    min_radius = 0
    new_width, new_height = max_radius * 2, max_radius * 2  
    # Calculate the center and max radius, and create an empty output image  
    center = new_height // 2, new_width // 2
    polar_img = np.ones((new_height, new_width, 3), dtype=np.uint8)  * 255
  

    # add the ending angle to 1.5 pi
    
  
    # Calculate the polar coordinates for each pixel in the destination image  
    for y in tqdm(range(new_height)):  
        for x in range(new_width):  
            dx, dy = x - center[0], y - center[1]  
            r = np.sqrt(dx * dx + dy * dy)  
            theta = np.arctan2(dy, dx) 
            # map the theta from -pi to pi to 0 to 2pi
            theta = theta + pi
            # scale the theta from 0 to 2pi to 0 to 1
            theta = theta / (1.5*pi)
            # so the x corresponds to the theta, y corresponds to the radius

            
            if theta > 1:
                continue
            else:
                x_orig = int(theta * width)
            y_orig = int(r) - min_radius
            if x_orig < 0 or x_orig >= width or y_orig < 0 or y_orig >= height:
                continue
            else:
                polar_img[y, x] = img[y_orig, x_orig]

    

     
  
    # Save the curved image  
    cv2.imwrite(output_path, polar_img)  
   
  
# Example usage  
image_path = "canvas.png"  
output_path = "curved_image_circle.png"  
#curve_image_circular(image_path, output_path)


# create another plot where it is a circular bar plot but only have 90 degree
import numpy as np  
import matplotlib.pyplot as plt  
  
def circular_barplot(data, labels, min_radius, max_radius, output_path):  
    # Normalize the data to fit between min_radius and max_radius  
    data = np.array(data)  
    data_normalized = min_radius + (max_radius - min_radius) * (data - data.min()) / (data.max() - data.min())  
  
    # Convert angles to radians  
    angles = np.linspace(0, np.pi / 2, len(data))  
  
    # Set up the polar plot  
    fig, ax = plt.subplots(subplot_kw={'polar': True})  
  
    # Remove the grid  
    ax.grid(False)  
  
    # Plot the bars with different colors  
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))  
    bars = ax.bar(angles, data_normalized, width=0.05, bottom=0, color=colors, alpha=0.7)  
  
    # Set the labels for the bars  
    ax.set_xticks(angles)  
    ax.set_xticklabels(labels)  
  
    # Set the y-axis label  
    ax.set_ylabel('Values')  
  
    # Save the circular barplot  
    plt.savefig(output_path)  
  
# Example usage  
data = [10, 20, 15, 5, 12]  
labels = ['A', 'B', 'C', 'D', 'E']  
min_radius = 5  
max_radius = 20  
output_path = "circular_barplot.png"  
circular_barplot(data, labels, min_radius, max_radius, output_path)  


