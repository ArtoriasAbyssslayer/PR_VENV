import cv2 as cv
#img = cv.imread("C:/Users/harry/Pictures/wallpapers/pf.jpg")

# application of gaussian blurring

#blur = cv.GaussianBlur(img, (7, 7))


img2 = cv.imread('C:/Users/harry/Pictures/ParanoidReverb.jpg')

blur2 = cv.medianBlur(img2,5)

cv.imwrite("C:/Users/harry/Pictures/ParanoidReverbReducedNoise.jpg",img2)


#Edge Casacade
canny = cv.Canny(img2, 135,190)
cv.imshow("Canny Edges", canny)
cv.waitKey(0)
