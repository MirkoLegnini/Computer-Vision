import cv2 
from matplotlib import pyplot as plt
import numpy as np



##########################################
## Task 1
##########################################

##Function declaration

def linear_stretching(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    transformed_img = 255./(max_value-min_value)*(img-min_value)
    print("Output image type: ", transformed_img.dtype)
    return transformed_img

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

show_images=True

for i in range (3):    
    img=cv2.imread(f"first task/C0_00000{i+1}.png", cv2.IMREAD_GRAYSCALE)
    # img_median=cv2.medianBlur(np.copy(img), 5)
    img_blurred = cv2.GaussianBlur(np.copy(img), [0,0], 2, None, 2) #gaussian filtering, sigma fine tuned
    if show_images: 
        plt.imshow(img, cmap='gray')
        plt.show()
    thresh, thresh_img= cv2.threshold(img_blurred, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #threshold without transformation
    gamma_img=exponential_operator(np.copy(img_blurred), 1/4)
    hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])
    #plt.stem(hist)
    #plt.show()
    max_value = find_percentile_value(hist, 70)
    min_value = find_percentile_value(hist, 10)
    transformed_img = linear_stretching(gamma_img, max_value, min_value).astype(np.uint8) #light transformations for contrast
    linear_stretched_img = linear_stretching(np.copy(img_blurred), max_value, min_value).astype(np.uint8) #as above w/out exp
    hist_st,bins = np.histogram(transformed_img.flatten(),256,[0,256])
    if show_images:
        plt.imshow(transformed_img,cmap='gray',vmin=0,vmax=255)
        plt.show()
    # plt.stem(hist_st)
    # plt.show()
    thresh, thres_transformed_image= cv2.threshold(transformed_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #thresholding on transformed image
    if show_images:
        # plt.subplot(1,2,1)
        # #plt.imshow(thresh_img, cmap='gray')
        # plt.subplot(1,2,2)
        plt.imshow(thres_transformed_image, cmap='gray')
        plt.show()

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # filled_img = cv2.morphologyEx(thres_transformed_image, cv2.MORPH_CLOSE, kernel) #closing to improve borders
    # filled_img=cv2.medianBlur(np.copy(img), 5) # bello ma no
    filled_img=cv2.medianBlur(np.copy(thres_transformed_image), 13)
    if show_images:
        plt.imshow(filled_img, cmap='gray')
        plt.show()


    img_color=cv2.imread(f"first task/C1_00000{i+1}.png") # open color images
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) #convert to right color map
    color_mask = np.concatenate([[filled_img]]*3, axis = 0) #create color mask 
    color_mask = np.moveaxis (color_mask, 0, -1)
    img_color_masked = cv2.bitwise_and(color_mask, img_color)
    if show_images:
        plt.subplot(1,2,1)
        plt.imshow(img_color)
        plt.subplot(1,2,2)
        plt.imshow(img_color_masked)
        plt.show()


    params = cv2.SimpleBlobDetector_Params() #create parameters for blob detection


    params.filterByArea = False
    params.minRepeatability = 1
    params.minThreshold = 0
    params.maxThreshold = 20
    params.minDistBetweenBlobs = 4

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(filled_img) #find key points for blobs

    if show_images and np.size(keypoints)!=0:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(img_color_masked, keypoints, blank, (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(blobs)
        plt.show()
    points = cv2.KeyPoint_convert(keypoints)
    for pt in points: # flood fill eventual holes in the apple
        cv2.floodFill(filled_img, None, tuple(pt.astype(int)), 255)


    if show_images: 
        plt.imshow(filled_img, cmap='gray')
        plt.show()


    filled_color_mask = np.concatenate([[filled_img]]*3, axis = 0)
    filled_color_mask = np.moveaxis (filled_color_mask, 0, -1)

    img_color_masked = cv2.bitwise_and(filled_color_mask, img_color)
    if show_images: 
        plt.imshow(img_color_masked)
        plt.show()

    img_gray_masked = cv2.bitwise_and(filled_img, linear_stretched_img.astype(np.uint8))
    img_blurred_masked = cv2.GaussianBlur(np.copy(img_gray_masked), [0,0], 0.7, None, 0.7)
    canny_threshold_high=120
    canny_threshold_low=10
    edges = cv2.Canny(img_blurred_masked, canny_threshold_low, canny_threshold_high)
    edges_kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    filled_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edges_kernel) #closing to improve borders
    # img_color_masked[:,:,2]=cv2.bitwise_or(img_color_masked[:,:,2], edges)
    # plt.imshow(img_color_masked)
    img_edged=cv2.bitwise_or(img, filled_edges)
    plt.imshow(img_edged, cmap="gray")
    plt.show()



##########################################
## Task 2
##########################################




##########################################
## Task 3
##########################################
