import cv2 
from matplotlib import pyplot as plt


print ('MELE')
## Aprire le immagini


img=cv2.imread("first task/C0_000001.png", cv2.IMREAD_GRAYSCALE)

print("this worked")
print(img);
plt.imshow(img, cmap='gray')
plt.show()

##########################################
## Task 1
##########################################


### 1) Threshold the image 
# 1.1) create_histogram
# 1.2) threshold using otsu's
# 1.3) show partial results


### 2) Fill the blobs
# 2.1)



##########################################
## Task 2
##########################################




##########################################
## Task 3
##########################################