import cv2
import numpy as np
from os import walk, remove
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import urllib.request
import requests
import difflib
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Normalization, RandomRotation, RandomZoom
from keras.utils import image_dataset_from_directory, img_to_array
from keras.callbacks import LearningRateScheduler
from os.path import exists
from tensorflow import expand_dims

class Player:
    def __init__(self):
        pass

    def toJson(self):
        return {
            'role': self.role,
            'username': self.username,
            'champion': self.champion,
            'lvl': self.lvl,
            'kda': self.kda,
            'dmg': self.dmg,
            'gold': self.gold,
        }


class ReadDigitModel:
    def __init__(self, modelPath = './models/readDigitModel.h5', epochs = 10, datasetPath = './digitDataset'):
        
        self.modelPath = modelPath if (modelPath != None and modelPath != '') else './models/readDigitModel.h5'
        self.datasetPath = datasetPath

        if(exists(self.modelPath)):
            self.loadModel()
        else:
            self.epochs = epochs
            self.createModel()
            self.trainModel()
            self.saveModel()
            
    def createModel(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size = 4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(11, activation='softmax'))

        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model

    def trainModel(self):
        train_ds = image_dataset_from_directory(
                                                self.datasetPath,
                                                labels='inferred',
                                                color_mode='grayscale',
                                                validation_split=0.2,
                                                subset="training",
                                                label_mode="categorical",
                                                seed=42,
                                                image_size=(28, 28),
                                                batch_size=32)

        val_ds = image_dataset_from_directory(
                                            self.datasetPath,
                                            labels='inferred',
                                            color_mode='grayscale',
                                            validation_split=0.2,
                                            subset="validation",
                                            label_mode="categorical",
                                            seed=42,
                                            image_size=(28, 28),
                                            batch_size=32)

        data_augmentation = Sequential([
            Normalization(),
            RandomRotation(0.1),
            RandomZoom (0.1)
        ])

        train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y))
            
        val_ds = val_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y))

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

        self.model.fit(train_ds, epochs=self.epochs, validation_data=val_ds, callbacks=[annealer])

    def saveModel(self):
        self.model.save(self.modelPath)

    def loadModel(self):
        self.model = load_model(self.modelPath)

    def read(self, img):
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)  # Create batch axis
        digit = int(np.argmax(self.model.predict(img_array, verbose = 0), axis=-1))
        if(digit == 2):
            digit = "/"
        elif digit > 2:
            digit -= 1
        
        return digit

# DETECT THE RED VERTICAL LINE (ENEMY TEAM) AND CROP THE LEFT PART OF THE IMAGE
def cropLeftPart(image):
    lower_red = np.array([170,50,50])
    upper_red = np.array([360,250,250])
    img_hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsl, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    detect_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # filter by lenght and get the one that is more at the left side of the image
    cnts = filter(lambda item: cv2.contourArea(item) > 100, cnts)
    cnts = sorted(cnts, key=lambda item1: item1[0][0][0])
    x, y, w, h = cv2.boundingRect(cnts[0])
    
    # crop the left part of the image
    return image[:, x + w:], y

# DETECT IF THERE IS FRIENDLIST (DILATING THE IMAGE VERTICALLY AND SEARCHING FOR A SEPARATED PART) AND CROP IT
def cropFriendList(image):
    image_enhanced = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    thresh = 200
    img_binary = cv2.threshold(image_enhanced, thresh, 255, cv2.THRESH_BINARY)[1]

    #dilate the image vertically to find the diferents parts of the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    image_dilated = cv2.dilate(img_binary, kernel, iterations=25)

    #find the contours of the image
    cnts = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #delete contours that are not useful for this function
    cnts = filter(lambda item: filterVerticalRightSide(item, img_binary), cnts)
    #sort from right to left
    cnts = sorted(cnts, key=lambda item1: item1[0][0][0], reverse=True)
    if(len(cnts) > 0):
        x, y, w, h = cv2.boundingRect(cnts[0])
        return image[:, 0:x - 10] #image without friend list
    
    return image

#DETECT IF THE IMAGE HAS TOP (VICTORY - DEFEAT) AND BOTTOM (PLAY AGAIN) AND CROP IT
def cropTopBottom(image, team2_y):
    image_enhanced = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    thresh = 250
    im_bw = cv2.threshold(image_enhanced, thresh, 255, cv2.THRESH_BINARY)[1]

    #dilate the image horizontally to find the diferents rows (usually are the top part, players and bottom part)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    image_dilated = cv2.dilate(im_bw, kernel, iterations=100)
    
    #dilate once vertically to avoid small parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    image_dilated = cv2.dilate(image_dilated, kernel, iterations=1)
    
    #find the different rows
    cnts = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #filter small rows
    cnts = filter(lambda item: filterMinVertical(item, im_bw), cnts)
    cnts = filter(lambda item: filterMinHorizontal(item), cnts)

    
    #sort from top to bottom
    cnts = sorted(cnts, key=lambda item1: item1[0][0][1])
        

    top = None
    bottom = None
    if(len(cnts) > 0):
        foundTop = False
        foundBottom = False
        cnts2 = cnts[:]
        numCntsTeam1 = len(list(filter(lambda item: filterTeam1(item, team2_y), cnts2)))

        victory_image = None
        #find the top player (if it has a lot of space between the countor and the previous one and is on the top of the image)
        while not foundTop:
            if(numCntsTeam1 > 5):
                cnts2.pop(0)
                numCntsTeam1 -= 1
                continue
            x, y, w, h = cv2.boundingRect(cnts2[0])
            x2, y2, w2, h2 = cv2.boundingRect(cnts2[1])
            x3, y3, w3, h3 = cv2.boundingRect(cnts2[2])
            if(abs((y3 - (y2 + h2)) - (y2 - (y + h))) < image.shape[0] / 40):
                victory_image = image[0:y, :]
                top = y
                foundTop = True
            elif((y2 - (y + h) > image.shape[0] / 20) and ((y3 - (y2 + h2)) <= (y2 - (y + h)))):
                victory_image = image[0:y + h, :]
                top = y2
                foundTop = True
            elif(y > image.shape[0] / 10): 
                top = y
                foundTop = True
            else:
                cnts2.pop(0)
            if(len(cnts2) < 2):
                top = 0
                foundTop = True

        cnts2 = cnts[:]
        numCntsTeam2 = len(list(filter(lambda item: filterTeam2(item, team2_y), cnts2)))
        #find the bottom player (if it has a lot of space between the countor and the previous one and is on the bottom of the image)
        while not foundBottom:
            if(numCntsTeam2 > 5):
                numCntsTeam2 -= 1
                cnts2.pop(-1)
                continue
                
            x, y, w, h = cv2.boundingRect(cnts2[-1])
            x2, y2, w2, h2 = cv2.boundingRect(cnts2[-2])
            x3, y3, w3, h3 = cv2.boundingRect(cnts2[-3])
            if (abs((y - (y2 + h2)) - (y2 - (y3 + h3))) < image.shape[0] / 40):
                bottom = y + h
                foundBottom = True
            elif((y - (y2 + h2) > image.shape[0] / 20) and ((y - (y2 + h2)) >= (y2 - (y3 + h3)))):
                bottom = y2 + h2
                foundBottom = True
            elif(y < image.shape[0] * 0.75):
                bottom = y + h
                foundBottom = True
            else:
                cnts2.pop(-1)
            if(len(cnts2) < 2):
                bottom = image.shape[0]
                foundBottom = True
        
        image = image[top:bottom, :]

        # if the image is too wide it can contain the friend list (the previous function may have failed)
        height = image.shape[0]
        width = image.shape[1]
        if(width / height > 2.25):
            image = image[:, 0: int(2.25 * height)]


        return image, victory_image
    else:
        return None, None

#DETECT EACH PLAYER ROW AND CROP IT
def getPlayerImages(image):
    image_enhanced = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    thresh = 250
    im_bw = cv2.threshold(image_enhanced, thresh, 255, cv2.THRESH_BINARY)[1]

    #dilate the image horizontally to find each player row
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilate = cv2.dilate(im_bw, kernel, iterations=100)
    #dilate once vertically to avoid small parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilate = cv2.dilate(dilate, kernel, iterations=1)

    #find the different rows
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #filter small rows
    cnts = filter(lambda item: filterMinVertical(item, im_bw), cnts)
    #sort from top to bottom
    cnts = sorted(cnts, key=lambda item1: item1[0][0][1])

    imgs = []
    #if has detected the 10 players, crop each player row
    if(len(cnts) == 10):
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            imgs.append(image[y:y + h + 5, :])
    #else find the mid gap and crop each player individually (knowing that each part has 5 players)
    else:
        x, y, w, h = cv2.boundingRect(cnts[0])
        x2, y2, w2, h2 = cv2.boundingRect(cnts[-1])

        i = 0
        maxSeparation = 0
        cntTop = None
        cntBottom = None
        #get the contourns with bigger gap between them
        for i in range(0, len(cnts) - 1):
            x3, y3, w3, h3 = cv2.boundingRect(cnts[i])
            x4, y4, w4, h4 = cv2.boundingRect(cnts[i + 1])
            if(y4 - (y3 + h3) > maxSeparation):
                maxSeparation = y4 - (y3 + h3)
                cntTop = cnts[i]
                cntBottom = cnts[i + 1]
            
        #crop each player row
        if(cntTop is not None and cntBottom is not None):
            x4, y4, w4, h4 = cv2.boundingRect(cntBottom)
            height = (y2 + h2 - y4) / 5 + 3
            for i in range(0, 10):
                y5 = y2 - height * i
                if(i > 4):
                    y5 -= maxSeparation / 2
                imgs.append(image[int(y5):int(y5 + height), :])
                imgs.reverse()
    return imgs

#GET THE CONTORNS THAT HAVE A HEIGHT > IMAGE.HEIGHT / 2, ITS WIDTH > IMAGE.WIDTH / 5 AND ARE IN THE RIGHT PART OF THE IMAGE
def filterVerticalRightSide(cnt, image):
    x, y, w, h = cv2.boundingRect(cnt)
    if h > (image.shape[0] / 2) and w < (image.shape[1] / 5) and x > (image.shape[1] * 0.75):
        return True
    return False

#ALL COUNTORNS HAVE TO BE HIGHER THAN THE IMAGE HEIGHT / 30
def filterMinVertical(cnt, image):
    x, y, w, h = cv2.boundingRect(cnt)
    if h > (image.shape[0] / 45):
        return True
    return False

#ALL COUNTORNS HAVE TO BE TOUCHING LEFT OF SCREEN
def filterMinHorizontal(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x < 10

#GET ALL CONTOURNS THAT BELONG TO TEAM 1 (OR ABOVE)
def filterTeam1(cnt, team2_y):
    x, y, w, h = cv2.boundingRect(cnt)
    if ((y < team2_y - 20) and ((y + h) < team2_y - 20)):
        return True
    return False

#GET ALL CONTOURNS THAT BELONG TO TEAM2 (OR BELOW)
def filterTeam2(cnt, team2_y):
    x, y, w, h = cv2.boundingRect(cnt)
    if y > team2_y - 20:
        return True
    return False

#READ SMALL BLOCK OF NUMBERS
def readLvl(image):
    data = pytesseract.image_to_string(image, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789')
    return data

#READ TEXT
def readTextUsernameAndChampion(image):
    data = pytesseract.image_to_string(image, config='--oem 3 --psm 7')
    return data

#READ A BIGGER BLOCK OF NUMBERS THAN READLVL
def readGoldAndDmg(image):
    data = pytesseract.image_to_string(image, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
    return data

#READ SINGLE DIGIT OF KDA (NUMBERS & /)
def readSingleDigit(image):
    data = pytesseract.image_to_string(image, config='--psm 10 -c tessedit_char_whitelist=0123456789/')
    return data

#READ WHOLE KDA (NUMBERS & /)
def readKDA(image):
    data = pytesseract.image_to_string(image, config='--psm 7 -c tessedit_char_whitelist=0123456789/')
    return data

#GET THE ELEMENTS COORDS (LVL, USERNAME, KDA, DMG, GOLD) TO DO THIS WE TRY TO FIND THE ELEMENT IN EACH IMAGE AND THEN ONLY GET THE COORDS THAT HAVE A RELATION BETWEEN (TO AVOID ERRORS)
def getCoords(images, element, x_crop = None):
    images_cropped = []
    #if we want to get the username we have to crop the left part of the image (the lvl part)
    if element == "username":
        if(x_crop == None):
            raise ValueError('Something went wrong')
        for image in images:
            image = images_cropped.append(image[:, x_crop:])
    #if we want to get the dmg or kda we have to crop the right part of the image (gold + dmg(only getting kda))
    elif element == 'dmg' or element == 'kda':
        if(x_crop == None):
            raise ValueError('Something went wrong')
        for image in images:
            images_cropped.append(image[:, :x_crop])
    else:
        images_cropped = images[:]
    
    
    coords = [] #coords of the element that we want to get
    x, y, w, h = 0, 0, 0, 0

    thresh = 225 if (element == 'lvl' or element == 'username') else 175
    for image in images_cropped:
        #preprocess the image and dilate to get the text in one element when getting the contours
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        im_bw = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(im_bw, kernel, iterations=3)

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #filter small contours
        cnts = filter(lambda item: cv2.contourArea(item) > 300, cnts)
        #lvl and username are at the left part of the image
        if(element == 'lvl' or element == "username"):
            cnts = sorted(cnts, key=lambda item1: item1[0][0][0])
        #gold, dmg and kda are at the right part of the image
        elif(element == 'gold' or element == 'dmg' or element == 'kda'):
            cnts = sorted(cnts, key=lambda item1: item1[0][0][0], reverse=True)

        if(element == 'lvl' or element == 'username'):
            
            # figure(figsize=(100, 100), dpi=80)
            # imgplot = plt.imshow(image, cmap='gray')
            # plt.show()
            if(len(cnts) > 2):
                for i in range(0, 2):
                    x, y, w, h = cv2.boundingRect(cnts[i])
                    x2, y2, w2, h2 = cv2.boundingRect(cnts[i + 1])

                    if(element == 'username'):
                        if(h > image.shape[0] * 0.75):
                            continue
                        elif(h2 > image.shape[0] * 0.75):
                            break 
                        elif(x + w > image.shape[1] * 0.4):
                            break
                        elif(w2 > w and x2 + w2 < image.shape[1] * 0.4):
                            break

                    if(x2 - (x + w) > 40):
                        break
                    else:
                        w += x2 - (x + w) + w2
                        
            elif(len(cnts) == 1):
                x, y, w, h = cv2.boundingRect(cnts[0])
            else:
                continue

            if(element == 'lvl'):
                if(x > image.shape[1] / 7):
                    continue
                
                if((w / h) > 1.25):
                    x = int((x + w) - h * 1.25)
                    w = int(h * 1.25)
        else:
            if(len(cnts) > 1):
                x, y, w, h = cv2.boundingRect(cnts[0])
                if(element == 'gold' and (w - h < 10 or w / h < 1.5 or h > image.shape[0] * 0.85)):
                    x, y, w, h = cv2.boundingRect(cnts[1])
            else:
                continue
            
        coords.append([x, w])

    #get the coords that have at leat one coord close (looking x)
    i = 0
    numImages = 0
    finalCoords = []
    for x, w in coords:
        numImages = 0
        for y in range(i + 1, len(coords)):
            x2, w2 = coords[y]
            if(abs(x - x2) < 20):
                numImages += 1
            if(numImages == 5):
                finalCoords.append([x, w])
                break
        i += 1

    x, y, w, h = 0, 0, 0, 0
    if(element == 'lvl'):
        x, y, w, h = 0, 0, 0, 0
        for x2, w2 in finalCoords:
            x += x2
            w += w2
                                        
        if(len(finalCoords) != 0):
            x /= len(finalCoords)
            w /= len(finalCoords)
    else:
        for x2, w2 in finalCoords:
            x = x2 if x2 < x or x == 0 else x
            w = w2 if w2 > w else w
    
    return int(x), 0, int(w), 0

def preprocessImg(img, threshold = 150):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img2 = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    mask = img2 < threshold
    img2[mask] = 0
    img2 = cv2.copyMakeBorder(img2, 10,5,10,5, borderType=cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.equalizeHist(img2)
    img2 = cv2.bilateralFilter(img2,4,25, 25)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    img2 = cv2.equalizeHist(img2)
    img2 = 255 - img2
    _,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY)

    return img2

#GET THE IMAGE IN A SHAPE OF 28x28 (ONLY WORKS WITH SINGLE DIGITS)
def preprocessDigit(img):
    padding_height_bot = 0 if img.shape[0] >= img.shape[1] else int((img.shape[1] - img.shape[0]) / 2)
    padding_height_top = 0 if img.shape[0] >= img.shape[1] else int((img.shape[1] - img.shape[0]) / 2) + ((img.shape[1] - img.shape[0]) % 2)
    padding_width_left = 0 if img.shape[1] >= img.shape[0] else int((img.shape[0] - img.shape[1]) / 2)
    padding_width_right = 0 if img.shape[1] >= img.shape[0] else int((img.shape[0] - img.shape[1]) / 2) + ((img.shape[0] - img.shape[1]) % 2)

    img2 = cv2.copyMakeBorder(img, padding_height_top, padding_height_bot, padding_width_right, padding_width_left, borderType=cv2.BORDER_CONSTANT, value=255)
    img2 = cv2.resize(img2, (26, 26))

    img2 = cv2.copyMakeBorder(img2, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=255)


    return img2

def getChampions():
    versions = requests.get('https://ddragon.leagueoflegends.com/api/versions.json')
    champions_array = []
    champions_ids = []
    if(versions.status_code == 200):
        latest_version = versions.json()[0]
        champions = requests.get('https://ddragon.leagueoflegends.com/cdn/' + latest_version + '/data/en_US/champion.json')
        if(champions.status_code == 200):
            for champion in champions.json()['data']:
                champions_array.append(champions.json()['data'][champion]['name'])
                champions_ids.append(champions.json()['data'][champion]['id'].lower())
        else:
            raise ValueError('Something went wrong')    
    else:
        raise ValueError('Something went wrong')  

    return champions_array, champions_ids

#GET KDA STRING FROM UNPROCESED IMG
def getKDAFromImage(kda_image, digitModel):
    #kda was not reading as good as the other ones, in this case we have divided into digits and read individually and if
    #it fails, we read the whole text
    kda_image = preprocessImg(kda_image)

    #find each digit
    contours, hierarchy = cv2.findContours(kda_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    #check that is not a contour inside a digit
    for y in range(len(contours)):
        if hierarchy[0,y,3] == 0:
            cnts.append(contours[y])
    #sort from left to right
    cnts = sorted(cnts, key=lambda item1: item1[0][0][0])
    kda_string = ''
    for cnt in cnts:
        x_digit, y_digit, w_digit, h_digit = cv2.boundingRect(cnt)
        img_digit = kda_image[y_digit:y_digit+h_digit, x_digit:x_digit+w_digit]
        #add border to the digit to make it easier to read
        img_digit = preprocessDigit(img_digit)

        digit = str(digitModel.read(img_digit))
        kda_string += digit
    
    kda_string2 = kda_string
    #it does not have exactly 2 / it means that there is an error in our data, so we read the whole text
    if(kda_string.count('/') != 2 ):
        kda_string = readKDA(kda_image).replace("\n", "")

    kda_split = kda_string.split('/')
    if(len(kda_split[0]) == 0):
        kda_split[0] = kda_string2.split('/')[0]
    if(len(kda_split[-1]) == 0):
        kda_split[-1] = kda_string2.split('/')[-1]
    
    kda_string = '/'.join(kda_split)
    return kda_string

#ALL COUNTORNS HAVE TO BE HIGHER THAN THE IMAGE HEIGHT / 30
def filterNonDigits(cnt, cnts):
    avg_h = sum(list(map(lambda temp_cnt: cv2.boundingRect(temp_cnt)[3], cnts))) / len(cnts)
    avg_h *= 0.9

    x, y, w, h = cv2.boundingRect(cnt)
    if h > avg_h:
        return True
    return False

#GET DMG/GOLD STRING FROM UNPROCESED IMG
def getDmgGoldLvlFromImage(img, digitModel):
    img = preprocessImg(img)

    #find each digit
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    #check that is not a contour inside a digit
    for y in range(len(contours)):
        if hierarchy[0,y,3] == 0:
            cnts.append(contours[y])
    cnts2 = list(cnts)

    #kda and gold sometimes have non digits characters, delete them by filtering only the big cnts
    cnts = filter(lambda item: filterNonDigits(item, cnts2), cnts)

    #sort from left to right
    cnts = sorted(cnts, key=lambda item1: item1[0][0][0])

    return_string = ''
    for cnt in cnts:
        x_digit, y_digit, w_digit, h_digit = cv2.boundingRect(cnt)
        img_digit = img[y_digit:y_digit+h_digit, x_digit:x_digit+w_digit]
        #add border to the digit to make it easier to read
        img_digit = preprocessDigit(img_digit)

        digit = str(digitModel.read(img_digit))
        return_string += digit

    return return_string
    

if __name__ == '__main__':
    # req = urllib.request.urlopen('https://storage.googleapis.com/esportslink-imges/posts/IMG2.png')
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # image = cv2.cvtColor(cv2.imdecode(arr, -1), cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(cv2.imread(f'./input/d89pKUgVrma4B3UfO941GeKzro4jtBdGnzLVDY0i.png'), cv2.COLOR_BGR2RGB)
    #crop left part (0 -> start of players stats)
    image, team2_y = cropLeftPart(image)
    #Crop the friends list (if exists)
    image = cropFriendList(image)
    #Crop the top (victory - defeat) and bottom (play again) part (if they exist)
    image, victoryImage = cropTopBottom(image, team2_y) #image is only the part of the players now

    #if error
    if(len(image) == None):
        raise ValueError('Something went wrong')
    
    #Get each player row (with all the stats)
    playerImages = getPlayerImages(image)
    

    i = 0

    x_lvl, y_lvl, w_lvl, h_lvl = getCoords(playerImages, 'lvl')
    x_username, y_username, w_username, h_username = getCoords(playerImages, 'username', x_lvl + w_lvl)
    x_username += (x_lvl + w_lvl)
    x_gold, y_gold, w_gold, h_gold = getCoords(playerImages, 'gold')
    x_dmg, y_dmg, w_dmg, h_dmg = getCoords(playerImages, 'dmg', x_gold - 5)
    x_kda, y_kda, w_kda, h_kda = getCoords(playerImages, 'kda', x_dmg - 5)
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    players = []
    champions_array, champions_ids = getChampions()

    digitModel = ReadDigitModel()
    for playerRow in playerImages:
        player = Player()
        player.role = roles[i]
        
        username_image = playerRow[0:int(playerRow.shape[0] * 0.6), x_username:x_username+w_username]
        username_image = preprocessImg(username_image)
        player.username = readTextUsernameAndChampion(username_image).replace("\n", "")

        champion_image = playerRow[int(playerRow.shape[0] * 0.5):, x_username:x_username+w_username]
        champion_image = preprocessImg(champion_image, 75)
        if(len(champions_array) > 0):
            champion_name = readTextUsernameAndChampion(champion_image).replace("\n", "")
            champion_search = difflib.get_close_matches(champion_name, champions_array)
            player.champion = champion_search[0] if len(champion_search) > 0 else 'unknown'
            if(player.champion == 'unknown'):
                champion_search = difflib.get_close_matches(champion_name, champions_ids)
                player.champion = champions_array[champions_ids.index(champion_search[0])] if len(champion_search) > 0 else 'unknown'
        else:
            player.champion = champion_name

        player.lvl = getDmgGoldLvlFromImage(playerRow[:, x_lvl:x_lvl+w_lvl], digitModel)

        player.gold = getDmgGoldLvlFromImage(playerRow[0:int(playerRow.shape[0] * 0.55), (x_gold - 5):(x_gold + w_gold + 5)], digitModel)

        player.dmg = getDmgGoldLvlFromImage(playerRow[0:int(playerRow.shape[0] * 0.55), (x_dmg - 5):(x_dmg + w_dmg + 5)], digitModel)

        player.kda = getKDAFromImage(playerRow[0:int(playerRow.shape[0] * 0.55), (x_kda - 5):(x_kda + w_kda + 5)], digitModel)
        
        players.append(player.toJson())
        
        i += 1
        i %= 5
    
    print(players)

        

