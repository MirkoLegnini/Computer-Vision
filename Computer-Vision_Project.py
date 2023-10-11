import cv2 
from matplotlib import pyplot as plt
import numpy as np


print ('MELE')
## Aprire le immagini

img=cv2.imread("first task/C0_000003.png", cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, [0,0], 2, None, 2)

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

def exponential_operator(img, r):
    exp_img = ((img/255)**r) *255
    return exp_img



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
max_value = find_percentile_value(hist, 60)
min_value = find_percentile_value(hist, 10)
exp_stretched_img=exponential_operator(np.copy(img), 0.5)
linear_stretched_img = linear_stretching(exp_stretched_img, max_value, min_value)

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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
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

params = cv2.SimpleBlobDetector_Params()


params.filterByArea = True
params.minArea = 1
params.maxArea = 1000000000000
params.minRepeatability = 1
params.minThreshold = 0
params.maxThreshold = 20
params.minDistBetweenBlobs = 4

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(filled_img)

blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(img_color_masked, keypoints, blank, (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(blobs)
plt.show()

points = cv2.KeyPoint_convert(keypoints)
print("\n blob centroids: ", points)

for keypoint in keypoints:
    print(keypoint.size)



for pt in points:
    cv2.floodFill(filled_img, None, tuple(pt.astype(int)), 255)


plt.imshow(filled_img, cmap='gray')
plt.show()


tensor = np.concatenate([[filled_img]]*3, axis = 0)
tensor = np.moveaxis (tensor, 0, -1)

print(np.shape(tensor))

img_color_masked = cv2.bitwise_and(tensor, img_color)
plt.subplot(1,2,2)
plt.imshow(img_color_masked)
plt.show()
##########################################
## Task 2
##########################################




##########################################
## Task 3
##########################################
