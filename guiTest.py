import selenium
import cv2 as cv
import numpy as np
import os
import imutils

from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By

def save_screenshot(driver: webdriver.Chrome, path) -> None:
    original_size = driver.get_window_size()
    required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
    driver.set_window_size(required_width, required_height)
    driver.find_element(By.TAG_NAME, "body").screenshot(path)
    driver.set_window_size(original_size['width'], original_size['height'])
    
def findSubImg(img, subImg):    
    img = cv.imread(img)
    if img is not None:
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgEdged = cv.Canny(imgGray, 50, 200)
        template = treatTemplate(subImg)
        found = None
        
        try:
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing
                resized = imutils.resize(template, width = int((template.shape[1]) * scale))
                (tH, tW) = resized.shape[:2]
                cv.imshow("Resized", resized)
                cv.waitKey(300)
                # r = template.shape[1] / float(resized.shape[1])
                
                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                result = cv.matchTemplate(imgEdged, resized, cv.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
                print(maxLoc)
            
                # draw a bounding box around the detected region
                clone = np.dstack([imgEdged, imgEdged, imgEdged])
                cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                
                cv.imshow("Visualize", clone)
                cv.waitKey(300)
                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc)
            # unpack the bookkeeping variable and compute the (x, y) coordinates
            # of the bounding box based on the resized ratio
            (_, maxLoc) = found
            (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
            (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
            
            # draw a bounding box around the detected result and display the image
            cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)   
            print(startX, ", ", startY)
            cv.imshow("Image", img)
            cv.waitKey(5000)
        except:
            print("Image not Found!")
            
        cv.destroyAllWindows()
    else:
        return "Base image not found"
    
def treatTemplate(img):
    template = cv.imread(img, cv.IMREAD_UNCHANGED)
    
    trans_mask = template[:,:,3] == 0
    template[trans_mask] = [255, 255, 255, 255]
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    template = cv.Canny(template, 50, 200)
    return template


# Desktop
opts = ChromeOptions()
opts.add_argument("--window-size=1920,1080")

# # iPhone 5
# opts = ChromeOptions()
# opts.add_argument("--window-size=640,1136")

driver = webdriver.Chrome(options=opts)
driver.get("https://metrobi.com/")
# driver.get("https://deliver.metrobi.com/signin")

assert "Metrobi" in driver.title

# sets the path for the screenshot
path = os.getcwd() + '\Screenshots\Screenshot.png'
save_screenshot(driver, path)

#find the coordinates of the image 
findSubImg(path, (os.getcwd() + '\Logo\logo.png'))


assert "No results found." not in driver.page_source
driver.close()