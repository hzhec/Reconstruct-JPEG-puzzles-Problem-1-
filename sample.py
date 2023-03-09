import cv2
import numpy as np
import glob
import math
from PIL import Image

# Reading an image in default mode:
inputImage = [cv2.imread(file) for file in glob.glob("simple/*.jpg")]

cv2_img = cv2.cvtColor(inputImage[0], cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(cv2_img)
image1_size = pil_img.size
numberOfImages = len(inputImage)
sqrtNo = int(math.sqrt(numberOfImages))
new_image = Image.new('RGB',(sqrtNo*image1_size[0], sqrtNo*image1_size[1]), (250,250,250))
 
# Defining lower and upper bound BGR values
lower_red = np.array([0, 0, 200])
upper_red = np.array([20, 20, 255])

lower_blue = np.array([200, 0, 0])
upper_blue = np.array([255, 20, 20])
  
# Defining mask for detecting color
for img in inputImage:
    mask_red = cv2.inRange(img, lower_red, upper_red)
    mask_blue = cv2.inRange(img, lower_blue, upper_blue)

    # Find the contours on the image:
    contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ObjectCounter:
    red_counter = 0
    blue_counter = 0
    

    # Look for the outer bounding boxes:
    for r in contours_red:
        red_counter += 1
    
    column = red_counter - 1
    
    for b in contours_blue:
        blue_counter += 1
        
    row = blue_counter - 1


    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img_cvt = Image.fromarray(cv2_img)
    
    new_image.paste(pil_img_cvt,(column*image1_size[0],row*image1_size[1]))

new_image.save("simple.jpg", "JPEG")
new_image.show()

    


    
    

    
