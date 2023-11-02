import cv2 
from matplotlib import pyplot as plt
import numpy as np

################################################################################

# FUNCTION DECLARATION #

def linear_stretching_inner(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    transformed_img = 255./(max_value-min_value)*(img-min_value)
    print("Output image type: ", transformed_img.dtype)
    return transformed_img

def linear_stretching(img, hist, min_perc, max_perc):
    max_value = find_percentile_value(hist, max_perc)
    min_value = find_percentile_value(hist, min_perc)
    res = linear_stretching_inner(img, max_value, min_value).astype(np.uint8) # light transformations for contrast
    return res

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


def create_binary_mask(img, r, min_perc, max_perc):
    img_blurred = cv2.GaussianBlur(np.copy(img), [0,0], 2, None, 2)     # gaussian filtering, sigma fine tuned
    gamma_img = exponential_operator(np.copy(img_blurred), r)
    hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])

    # QUA PERCHE PER FARE IL LINEAR STRETCHING VIENE USATA LA IMMAGINE EXPONENTIAL E L HISTOGRAM DELLA BLURRED
    
    transformed_img = linear_stretching(gamma_img,hist, min_perc, max_perc)
    
    if show_images:
        plt.imshow(transformed_img,cmap = 'gray',vmin=0,vmax=255)
        plt.show()

    thresh, thres_transformed_image = cv2.threshold(transformed_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    #thresholding on transformed image

    if show_images:

        plt.imshow(thres_transformed_image, cmap='gray')
        plt.show()

    filled_img=cv2.medianBlur(np.copy(thres_transformed_image), 13)

    if show_images:
        plt.imshow(filled_img, cmap='gray')
        plt.show()

    params = cv2.SimpleBlobDetector_Params()            # create parameters for blob detection
    params.filterByArea = False
    params.minRepeatability = 1
    params.minThreshold = 0
    params.maxThreshold = 20
    params.minDistBetweenBlobs = 4
    params.filterByCircularity = False 
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(filled_img)             # find key points for blobs
    
    if show_images and np.size(keypoints)!=0:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(filled_img, keypoints, blank, (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(blobs)
        plt.show()

    points = cv2.KeyPoint_convert(keypoints)
    for pt in points:                                   # flood fill eventual holes in the apple
        cv2.floodFill(filled_img, None, tuple(pt.astype(int)), 255)
    
    if show_images: 
        plt.imshow(filled_img, cmap='gray')
        plt.show()
    
    return filled_img


def create_stats():
    flat_list=[]
    for i in range(25):
        img=cv2.imread(f"second task/stats/{i+1}.png")
        flat = img.reshape(-1,3).T
        flat_list.append(flat)
    flat_arr = np.concatenate(flat_list, -1)

    mean = np.mean(flat_arr, 1)
    print(mean)
    cov = np.cov(flat_arr)
    iconv = np.linalg.inv(cov)
    return mean, iconv

##########################################
## Task 1
##########################################

show_images = False
show_images_2 = True

for i in range (0):    
    ## Parameter declaration for light transformations
    min_perc=10
    max_perc=70
    r=0.25

    img = cv2.imread(f"first task/C0_00000{i+1}.png", cv2.IMREAD_GRAYSCALE)
    img_blurred = cv2.GaussianBlur(np.copy(img), [0,0], 2, None, 2)     # gaussian filtering, sigma fine tuned
    if show_images: 
        plt.imshow(img, cmap='gray')
        plt.show()

    binary_mask = create_binary_mask(np.copy(img), r, min_perc, max_perc)
    color_mask = cv2.merge([binary_mask, binary_mask, binary_mask])     # create color mask 
    img_masked = cv2.bitwise_and(color_mask, img)

    if show_images:
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(img_masked)
        plt.show()
    
    '''
    img_color=cv2.imread(f"first task/C1_00000{i+1}.png")               # open color images
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    filled_color_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
    img_color_masked = cv2.bitwise_and(filled_color_mask, img_color)
    if show_images: 
        plt.imshow(img_color_masked)
        plt.show()
    '''

    hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])
    linear_stretched_img = linear_stretching(np.copy(img_blurred),hist, min_perc, max_perc)

    img_gray_masked = cv2.bitwise_and(binary_mask, linear_stretched_img.astype(np.uint8))
    # img_blurred_masked = cv2.GaussianBlur(np.copy(img_gray_masked), [0,0], 0.7, None, 0.7)

    canny_threshold_high=120
    canny_threshold_low=10
    edges = cv2.Canny(img_gray_masked, canny_threshold_low, canny_threshold_high)
    edges_kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    filled_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edges_kernel)   # closing to improve borders
    
    img_edged=cv2.bitwise_or(img, filled_edges)
    plt.imshow(img_edged, cmap="gray")
    plt.show()

    # get contours from image
    contours, hierarchy = cv2.findContours(filled_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    img=cv2.merge([img, img, img])
    for i in range(len(contours)):
        opened =  hierarchy[0][i][2]<0
        if (not opened) and cv2.contourArea(contours[i])<20000:
            cv2.drawContours(img, contours, i, (0,255,0), 2)
    plt.imshow(img)
    plt.show()

##########################################
## Task 2
##########################################

flat_tot = []
img_xd = []
for j in range(2):
    img_xd.append(cv2.imread(f"second task/C1_00000{j+4}.png"))

width = min(img_xd[1].shape[1], img_xd[0].shape[1])
height = min(img_xd[1].shape[0], img_xd[0].shape[0])

image1 = cv2.resize(img_xd[1], (width, height))
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(img_xd[0], (width, height))
image2=cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

img_avg = (image1.astype(np.uint16) + image2.astype(np.uint16)) // 2
img_avg = img_avg.astype(np.uint8)

plt.subplot(1,3,1)
plt.imshow(image1)
plt.subplot(1,3,2)
plt.imshow(image2)
plt.subplot(1,3,3)
plt.title("Merged Images")
plt.imshow(img_avg)
plt.show()

#by looking at the merged images we trace a section where the pixels seem lighter

x1, y1 = 50, 35
x2, y2 = 70, 55

# x1, y1 = 165, 80
# x2, y2 = 195, 130

'''
for k in range(2):
    img = cv2.imread(f"second task/C0_00000{k+4}.png")
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(np.copy(img_bw), [0,0], 2, None, 2)
    hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])
    plt.stem(hist)
    plt.show
'''

for i in range(2):

    ## Parameter declaration for light transformations
    min_perc = 10
    max_perc = 70
    r=0.5

    img = cv2.imread(f"second task/C1_00000{i+4}.png")
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # we use colored image and change it in BW instead of using standard BW because of parallax
    
    img_blurred = cv2.GaussianBlur(np.copy(img_bw), [0,0], 2, None, 2)    # gaussian filtering, sigma fine tuned
    binary_mask = create_binary_mask(np.copy(img_bw), r, min_perc, max_perc)
    
    img_color = cv2.imread(f"second task/C1_00000{i+4}.png")              # open color images
    """
    img_color[:,:,2] = exponential_operator(img_color[:,:,2], 0.6).astype(np.uint8) #exponential stretch one color at a time in BGR
    img_color[:,:,1] = exponential_operator(img_color[:,:,1], 1.3).astype(np.uint8)
    #img_color[:,:,0] = exponential_operator(img_color[:,:,0], 0.5).astype(np.uint8)
    """
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)              # convert to right color map

    color_mask = cv2.merge([binary_mask, binary_mask, binary_mask])     # create color mask
    img_color_masked = cv2.bitwise_and(color_mask, img_color)

    """
    if show_images_2:
        plt.imshow(img_color_masked[:,:,2],cmap = 'gray')
        plt.title("V value")
        plt.show()
        
    thresh_hsv, thresh_hsv_img = cv2.threshold(img_color_masked[:,:,2], 190, 255, cv2.THRESH_BINARY)

    if show_images_2:
        plt.imshow(thresh_hsv_img, cmap = 'gray')
        plt.title("threshold on V value")
        plt.show()

    mean_mask = cv2.merge([thresh_hsv_img,thresh_hsv_img,thresh_hsv_img])
    img_masked_onV = cv2.bitwise_and(mean_mask, img_color_masked)

    if show_images_2:
        plt.imshow(img_masked_onV)
        plt.title("image masked for mean calculation")
        plt.show()

    medianed_img=cv2.medianBlur(np.copy(img_color), 11)
    med_img_masked = cv2.bitwise_and(color_mask, medianed_img)
    img_color_masked = med_img_masked
    """

    if show_images_2:
        plt.subplot(1,3,1)
        plt.title("BW Image")
        plt.imshow(img_bw, cmap="gray")
        plt.subplot(1,3,2)
        plt.title("Colored Image")
        plt.imshow(img_color)
        plt.subplot(1,3,3)
        plt.title("Masked Image")
        plt.imshow(img_color_masked)
        plt.show()

    # select the right section of the image

    img_color_masked = cv2.cvtColor(img_color_masked, cv2.COLOR_RGB2HSV)
    section = img_color_masked[y1:y2, x1:x2]
    img_sectioned = np.copy(section)

    # print('image section shape:', np.shape(img_sectioned))

    flat = img_sectioned.reshape(-1,3).T
    flat = flat[:, np.all(flat!=0, 0)]
    flat_tot.append(flat)
    flat_arr = np.concatenate(flat_tot, -1)

    mean = np.mean(flat_arr, 1)
    cov = np.cov(flat)
    iconv = np.linalg.inv(cov)


    mean, iconv = create_stats()
      
    img_flat = np.reshape(img_color_masked,(np.shape(img_color_masked)[0]*np.shape(img_color_masked)[1],3))
    diff = img_flat-mean
    diff[img_flat==0] = 0
    
    mahal = np.sqrt(np.multiply(np.dot(diff,iconv),diff).sum(-1))
    mahal_img = mahal.reshape(np.shape(img_color_masked)[0],np.shape(img_color_masked)[1])
    hist_mahal,bins_mahal = np.histogram(mahal_img.flatten(),256,[0,256])

    # print(diff)
    # print(np.shape(mahal))
    # print(np.shape(mahal_img))

    dist = np.sqrt(np.multiply(diff,diff)).sum(-1)
    dist_img = dist.reshape(np.shape(img_color_masked)[0],np.shape(img_color_masked)[1])
    hist_eucl,bins_eucl = np.histogram(dist_img.flatten(),256,[0,256])

    if show_images_2:
        plt.subplot(3,2,1)
        plt.title("Image Colored")
        plt.imshow(img_color)
        plt.subplot(3,2,3)
        plt.title("Mahalanobis Distance")
        plt.imshow(mahal_img, cmap="gray")
        plt.subplot(3,2,4)
        plt.title("Mahalanobis Histogram")
        plt.stem(hist_mahal)
        plt.subplot(3,2,5)
        plt.title("Euclidean Distance")
        plt.imshow(dist_img, cmap="gray")
        plt.subplot(3,2,6)
        plt.title("Euclidian Histogram")
        plt.stem(hist_eucl)
        plt.show()

    mahal_img_blurred = cv2.GaussianBlur(np.copy(mahal_img), [0,0], 1.5, None, 1.5)

    # remove background
    mahal_img_blurred = mahal_img_blurred.astype(float)
    black_pixels = (mahal_img_blurred == 0)
    mahal_img_blurred[black_pixels] = np.nan
   
    # applying exponential operator
    exp_img = exponential_operator(mahal_img_blurred, 0.45)
    exp_hist_mahal,bins_mahal = np.histogram(exp_img.flatten(),256,[0,256])

    plt.subplot(1,2,1)
    plt.title("Exp Image")
    plt.imshow(exp_img, cmap="gray")
    plt.subplot(1,2,2)
    plt.title("Exp Histogram")
    plt.stem(exp_hist_mahal)
    plt.show()

    # stretching linearly
    transformed_img = linear_stretching(np.copy(exp_img), exp_hist_mahal, 50, 80)
    stretch_hist_mahal,bins_mahal = np.histogram(transformed_img.flatten(),256,[0,256])

    plt.subplot(2,2,1)
    plt.title("Mahalanobis Original Image")
    plt.imshow(mahal_img, cmap="gray")
    plt.subplot(2,2,2)
    plt.title("Mahalanobis Original Histogram")
    plt.stem(hist_mahal)
    plt.subplot(2,2,3)
    plt.title("Stretched Mahalanobis Image")
    plt.imshow(transformed_img, cmap = 'gray')
    plt.subplot(2,2,4)
    plt.title("Stretched Mahalanobis Histogram")
    plt.stem(stretch_hist_mahal)
    plt.show()

    ret, thresholded_mahal_image = cv2.threshold(transformed_img.astype(np.uint8), 126, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, thresholded_mahal_image = cv2.threshold(mahal_img_blurred.astype(np.uint8), 22, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

    russet_mask = cv2.merge([thresholded_mahal_image, thresholded_mahal_image, thresholded_mahal_image])
    norusset_img = cv2.bitwise_and(russet_mask, img_color)
     
    plt.subplot(1,2,1)
    plt.imshow(img_color)
    plt.title("Image Colored")
    plt.subplot(1,2,2)
    plt.imshow(norusset_img)
    plt.title("Image With No Russet")
    plt.show()


##########################################
## Task 3
##########################################