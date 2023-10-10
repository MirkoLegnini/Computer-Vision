import cv2 
from matplotlib import pyplot as plt
import numpy as np


print ('MELE')
## Aprire le immagini

img=cv2.imread("first task/C0_000003.png", cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, [0,0], 1.5, None, 1.5)

plt.imshow(img, cmap='gray')
plt.show()

##########################################
## Task 1
##########################################


### 1) Threshold the image 

# 1.1) create_histogram
# 1.2) threshold using otsu's
# 1.3) show partial results

thresh, filtered_img= cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# adapt_img= cv2.adaptiveThreshold(img,  255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def linear_stretching(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    linear_stretched_img = 255./(max_value-min_value)*(img-min_value)
    print("Output image type: ", linear_stretched_img.dtype)
    return linear_stretched_img


def find_percentile_value(hist, percentile):
    s = 0
    idx = 0
    total_pixel = np.sum(hist)
    while(s < total_pixel*percentile/100):
        s += hist[idx]
        idx += 1
    return idx

hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.stem(hist)
plt.show()
max_value = find_percentile_value(hist, 50)
min_value = find_percentile_value(hist, 1)
linear_stretched_img=linear_stretching(np.copy(img), max_value,min_value)
hist_st,bins = np.histogram(linear_stretched_img.astype(np.uint8).flatten(),256,[0,256])
plt.imshow(linear_stretched_img,cmap='gray',vmin=0,vmax=255)
plt.show()
plt.title("Hist After Stretching")

# 1.4) Include "stem"

plt.stem(hist_st)
plt.show()
thresh, filtered_stretched_img= cv2.threshold(linear_stretched_img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(1,2,1)
plt.imshow(filtered_img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(filtered_stretched_img, cmap='gray')
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
filled_img = cv2.morphologyEx(filtered_stretched_img, cv2.MORPH_CLOSE, kernel)
plt.imshow(filled_img, cmap='gray')
plt.show()


img_color=cv2.imread("first task/C1_000003.png")
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img_color)

print(np.shape(filtered_stretched_img))
print(np.shape(img_color))

tensor = np.concatenate([[filled_img]]*3, axis = 0)
tensor = np.moveaxis (tensor, 0, -1)

print(np.shape(tensor))

img_color_masked = cv2.bitwise_and(tensor, img_color)
plt.subplot(1,2,2)
plt.imshow(img_color_masked)
plt.show()

#blob analysys scartando il primo blob (sfondo)

### 2) Fill the blobs
# 2.1)



##########################################
## Task 2
##########################################




##########################################
## Task 3
##########################################