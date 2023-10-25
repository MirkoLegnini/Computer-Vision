import cv2 
from matplotlib import pyplot as plt
import numpy as np



##########################################
## Task 1
##########################################

##Function declaration



def linear_stretching_inner(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    transformed_img = 255./(max_value-min_value)*(img-min_value)
    print("Output image type: ", transformed_img.dtype)
    return transformed_img

def linear_stretching(img, hist, min_perc, max_perc):
    max_value = find_percentile_value(hist, max_perc)
    min_value = find_percentile_value(hist, min_perc)
    res = linear_stretching_inner(img, max_value, min_value).astype(np.uint8) #light transformations for contrast
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


def create_binary_mask(img_blurred, r, min_perc, max_perc):
    gamma_img=exponential_operator(np.copy(img_blurred), r)
    hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])
    #plt.stem(hist)
    #plt.show()
    transformed_img=linear_stretching(gamma_img,hist, min_perc, max_perc);
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

    params = cv2.SimpleBlobDetector_Params() #create parameters for blob detection
    
    
    params.filterByArea = False
    params.minRepeatability = 1
    params.minThreshold = 0
    params.maxThreshold = 20
    params.minDistBetweenBlobs = 4
    params.filterByCircularity = False 
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(filled_img) #find key points for blobs
    
    if show_images and np.size(keypoints)!=0:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(filled_img, keypoints, blank, (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(blobs)
        plt.show()
    points = cv2.KeyPoint_convert(keypoints)
    for pt in points: # flood fill eventual holes in the apple
        cv2.floodFill(filled_img, None, tuple(pt.astype(int)), 255)
    
    
    if show_images: 
        plt.imshow(filled_img, cmap='gray')
        plt.show()
    

    return filled_img


show_images=True
show_images_2=True
if 0:
    for i in range (1):    
        ## Parameter declaration for light transformations
        min_perc=10
        max_perc=70
        r=0.25
    
        img=cv2.imread(f"first task/C0_00000{i+1}.png", cv2.IMREAD_GRAYSCALE)
        img_blurred = cv2.GaussianBlur(np.copy(img), [0,0], 2, None, 2) #gaussian filtering, sigma fine tuned
        if show_images: 
            plt.imshow(img, cmap='gray')
            plt.show()
        binary_mask=create_binary_mask(np.copy(img_blurred), r, min_perc, max_perc)
    
    
        img_color=cv2.imread(f"first task/C1_00000{i+1}.png") # open color images
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) #convert to right color map
        color_mask = cv2.merge([binary_mask, binary_mask, binary_mask]) #create color mask 
        img_color_masked = cv2.bitwise_and(color_mask, img_color)
        if show_images:
            plt.subplot(1,2,1)
            plt.imshow(img_color)
            plt.subplot(1,2,2)
            plt.imshow(img_color_masked)
            plt.show()
    
    
        
        filled_color_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
    
        img_color_masked = cv2.bitwise_and(filled_color_mask, img_color)
        if show_images: 
            plt.imshow(img_color_masked)
            plt.show()
    
    
    
        hist,bins = np.histogram(img_blurred.flatten(),256,[0,256])
        linear_stretched_img=linear_stretching(np.copy(img_blurred),hist, min_perc, max_perc)
        img_gray_masked = cv2.bitwise_and(binary_mask, linear_stretched_img.astype(np.uint8))
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
flat_tot=[]
for i in range(2):
    min_perc=10
    max_perc=70
    r=0.5
    img=cv2.imread(f"second task/C1_00000{i+4}.png")
    img_bw=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(np.copy(img_bw), [0,0], 2, None, 2) #gaussian filtering, sigma fine tuned
    if show_images: 
        plt.imshow(img, cmap='gray')
        plt.show()
    binary_mask=create_binary_mask(np.copy(img_blurred), r, min_perc, max_perc)
    img_color=cv2.imread(f"second task/C1_00000{i+4}.png") # open color images
    
    img_color[:,:,2] = exponential_operator(img_color[:,:,2], 0.5).astype(np.uint8) #exponential stretch one color at a time in BGR
    img_color[:,:,1] = exponential_operator(img_color[:,:,1], 1.5).astype(np.uint8)
    #img_color[:,:,0] = exponential_operator(img_color[:,:,0], 0.5).astype(np.uint8)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) #convert to right color map
    color_mask = cv2.merge([binary_mask, binary_mask, binary_mask]) #create color mask 
    img_color_masked = cv2.bitwise_and(color_mask, img_color)

    medianed_img=cv2.medianBlur(np.copy(img_color), 11)

    if show_images_2:
        plt.imshow(medianed_img)
        plt.title("meadianed image")
        plt.show()
    med_img_masked = cv2.bitwise_and(color_mask, medianed_img)
    #img_color_masked = med_img_masked
    if show_images_2:
        plt.subplot(1,2,1)
        plt.imshow(img_bw, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(img_color_masked)
        plt.show()

    flat = img_color_masked.reshape(-1,3).T
    flat=flat[:, np.all(flat!=0, 0)]

    flat_tot.append(flat)
flat_arr=np.concatenate(flat_tot, -1)
print(np.shape(flat_arr))
mean=np.mean(flat_arr, 1)
cov = np.cov(flat)
print("variance is: ", cov)
iconv = np.linalg.inv(cov)

for i in range(2):
    img=cv2.imread(f"second task/C1_00000{i+4}.png")
    min_perc=10
    max_perc=70
    r=0.5
    img_bw=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(np.copy(img_bw), [0,0], 2, None, 2) #gaussian filtering, sigma fine tuned
    if show_images: 
        plt.imshow(img, cmap='gray')
        plt.show()
    binary_mask=create_binary_mask(np.copy(img_blurred), r, min_perc, max_perc)
    
    img_color=cv2.imread(f"second task/C1_00000{i+4}.png") # open color images
    
    img_color[:,:,2] = exponential_operator(img_color[:,:,2], 0.5).astype(np.uint8) #exponential stretch one color at a time in BGR
    img_color[:,:,1] = exponential_operator(img_color[:,:,1], 1.5).astype(np.uint8)
    #img_color[:,:,0] = exponential_operator(img_color[:,:,0], 0.5).astype(np.uint8)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) #convert to right color map
    color_mask = cv2.merge([binary_mask, binary_mask, binary_mask]) #create color mask 
    img_color_masked = cv2.bitwise_and(color_mask, img_color)

    medianed_img=cv2.medianBlur(np.copy(img_color), 11)
    if show_images_2:
        plt.imshow(medianed_img)
        plt.title("meadianed image")
        plt.show()
    med_img_masked = cv2.bitwise_and(color_mask, medianed_img)
    #img_color_masked = med_img_masked
    if show_images_2:
        plt.subplot(1,2,1)
        plt.imshow(img_bw, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(img_color_masked)
        plt.show()
    # finding average image colour
    # arg = np.where(med_img_masked!=0,med_img_masked,np.nan)
    # print(arg)
    # mean_clr = np.nanmean(arg,(0,1))
    # print("mean is: ", mean_clr)
    # flat = img_color_masked.reshape(-1,3).T
    # flat=flat[:, np.all(flat!=0, 0)]
    # mean_2=np.mean(flat, (1))
    # print("mean is: ", mean_2)
    # #variance = np.cov(np.where(img_color_masked[:,:,0]!=0,img_color_masked,np[:,:,0].nan))
    # #variance = np.cov(np.where(img_color_masked[:,:,1]!=0,img_color_masked,np[:,:,1].nan))
    # #variance = np.cov(np.where(img_color_masked[:,:,2]!=0,img_color_masked,np[:,:,2].nan))
    # print("flat filtered=", flat)
    # print('shape=', np.shape(flat))
    # cov = np.cov(flat)
    # print("variance is: ", cov)
    # mean_img = np.zeros(np.shape(img_color))
    # print(mean_clr)

    # mean_img[:]=mean_2
    # print(np.shape(mean_img))
    # print(mean_img[0][0])
    # plt.imshow(mean_img.astype(np.uint8))
    # plt.show()
    # #print("color mask shape: ", np.shape(img_color_masked))
    # #print("mean shape: ", np.shape(mean_img))
    # #print(type(img_color_masked))
    # #covar, mean = cv2.calcCovarMatrix(np.where(flat!=0,flat, flat), None, cv2.COVAR_NORMAL | cv2.COVAR_COLS)
    # #print("mean with cv2 is: ", mean)
    # #print("covar with cv2 is: ", covar)

    # iconv = np.linalg.inv(cov)




    ##Actual computation of the distance


     
    img_flat = np.reshape(img_color_masked,(np.shape(img_color_masked)[0]*np.shape(img_color_masked)[1],3))
    diff=img_flat-mean
    diff[img_flat==0]=0
    print(diff)
    mahal = np.sqrt(np.multiply(np.dot(diff,iconv),diff).sum(-1))
    print(np.shape(mahal))
    mahal_img = mahal.reshape(np.shape(img_color_masked)[0],np.shape(img_color_masked)[1])
    plt.subplot(1,2,1)
    plt.imshow(img_color)
    plt.subplot(1,2,2)
    plt.imshow(mahal_img, cmap="gray")
    plt.title("mahalanobis distance")
    plt.show()
    
    hist,bins = np.histogram(mahal_img.flatten(),256,[0,256])
    plt.stem(hist)
    plt.show()

    dist = np.sqrt(np.multiply(diff,diff)).sum(-1)
    dist_img = dist.reshape(np.shape(img_color_masked)[0],np.shape(img_color_masked)[1])
    plt.subplot(1,2,1)
    plt.imshow(img_color)
    plt.subplot(1,2,2)
    plt.imshow(dist_img, cmap="gray")
    plt.title("euclidean distance")
    plt.show()

    hist,bins = np.histogram(dist_img.flatten(),256,[0,256])
    plt.stem(hist)
    plt.show()





##########################################
## Task 3
##########################################
