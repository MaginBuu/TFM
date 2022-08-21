import cv2
import numpy as np
from os import walk, remove
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import urllib.request
import requests
import difflib

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

# DETECT THE RED VERTICAL LINE (ENEMY TEAM) AND CROP THE LEFT PART OF THE IMAGE
def cropLeftPart(image):
    lower_red = np.array([170,50,50])
    upper_red = np.array([360,250,250])
    img_hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsl, lower_red, upper_red)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detect_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # filter by lenght and get the one that is more at the left side of the image
    cnts = filter(lambda item: cv2.contourArea(item) > 100, cnts)
    cnts = sorted(cnts, key=lambda item1: item1[0][0][0])
    x, y, w, h = cv2.boundingRect(cnts[0])
    
    # crop the left part of the image
    return image[:, x + w:]

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
def cropTopBottom(image):
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
    #sort from top to bottom
    cnts = sorted(cnts, key=lambda item1: item1[0][0][1])

    
    top = None
    bottom = None
    if(len(cnts) > 0):
        foundTop = False
        foundBottom = False
        cnts2 = cnts[:]
        victory_image = None
        #find the top player (if it has a lot of space between the countor and the previous one and is on the top of the image)
        while not foundTop:
            x, y, w, h = cv2.boundingRect(cnts2[0])
            x2, y2, w2, h2 = cv2.boundingRect(cnts2[1])
            if(y2 - (y + h) > image.shape[0] / 20):
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
        #find the bottom player (if it has a lot of space between the countor and the previous one and is on the bottom of the image)
        while not foundBottom:
            x, y, w, h = cv2.boundingRect(cnts2[-1])
            x2, y2, w2, h2 = cv2.boundingRect(cnts2[-2])
            if(y - (y2 + h2) > image.shape[0] / 20):
                bottom = y2 + h2
                foundBottom = True
            elif(y > image.shape[0] / 2):
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
    if h > (image.shape[0] / 30):
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
    for image in images_cropped:
        #preprocess the image and dilate to get the text in one element when getting the contours
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = 225
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
            if(len(cnts) > 2):
                for i in range(0, 2):
                    x, y, w, h = cv2.boundingRect(cnts[i])
                    x2, y2, w2, h2 = cv2.boundingRect(cnts[i + 1])
                    if(x2 - (x + w) > 50):
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

            else:
                continue

        coords.append([x, w])

    #get the coords that have at leat one coord close (looking x)
    i = 0
    finalCoords = []
    for x, w in coords:
        for y in range(i + 1, len(coords)):
            x2, w2 = coords[y]
            if(abs(x - x2) < 30):
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
    mask = img < threshold
    img[mask] = 0
    img2 = cv2.copyMakeBorder(img, 10,5,10,5, borderType=cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.equalizeHist(img2)
    img2 = cv2.bilateralFilter(img2,4,25, 25)
    img2 = cv2.resize(img2, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    img2 = cv2.equalizeHist(img2)
    img2 = 255 - img2
    _,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY)

    return img2


def preprocessImg2(img):
    img2 = cv2.copyMakeBorder(img, 5,5,5,5, borderType=cv2.BORDER_CONSTANT, value=255)
    img2 = cv2.equalizeHist(img2)
    img2 = cv2.bilateralFilter(img2,4,25, 25)
    img2 = cv2.resize(img2, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    img2 = cv2.equalizeHist(img2)
    _,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY)

    return img2


def getChampions():
    versions = requests.get('https://ddragon.leagueoflegends.com/api/versions.json')
    champions_array = []
    if(versions.status_code == 200):
        latest_version = versions.json()[0]
        champions = requests.get('https://ddragon.leagueoflegends.com/cdn/' + latest_version + '/data/en_US/champion.json')
        if(champions.status_code == 200):
            for champion in champions.json()['data']:
                champions_array.append(champions.json()['data'][champion]['name'])
        else:
            raise ValueError('Something went wrong')    
    else:
        raise ValueError('Something went wrong')  

    return champions_array

if __name__ == '__main__':
    # req = urllib.request.urlopen('https://storage.googleapis.com/esportslink-imges/posts/IMG2.png')
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # image = cv2.cvtColor(cv2.imdecode(arr, -1), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"./output/Item.png", image)
    image = cv2.cvtColor(cv2.imread(f'./input/img4.jpg'), cv2.COLOR_BGR2RGB)
    #crop left part (0 -> start of players stats)
    image = cropLeftPart(image)
    cv2.imwrite(f"./output/1.png", image)
    #Crop the friends list (if exists)
    image = cropFriendList(image)
    cv2.imwrite(f"./output/2.png", image)
    #Crop the top (victory - defeat) and bottom (play again) part (if they exist)
    image, victoryImage = cropTopBottom(image) #image is only the part of the players now
    cv2.imwrite(f"./output/3.png", image)

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
    champions_array = getChampions()

    for playerRow in playerImages:
        player = Player()
        player.role = roles[i]

        #crop the part of image that contains text and read it
        lvl_image = playerRow[:, x_lvl:x_lvl+w_lvl] 
        lvl_image = preprocessImg(lvl_image)
        player.lvl = readLvl(lvl_image).replace("\n", "")

        
        username_image = playerRow[0:int(playerRow.shape[0] * 0.5), x_username:x_username+w_username]
        username_image = preprocessImg(username_image)
        player.username = readTextUsernameAndChampion(username_image).replace("\n", "")
        cv2.imwrite(f"./output/Item{i}_username.png", username_image)

        champion_image = playerRow[int(playerRow.shape[0] * 0.5):, x_username:x_username+w_username]
        champion_image = preprocessImg(champion_image, 75)
        if(len(champions_array) > 0):
            champion_name = readTextUsernameAndChampion(champion_image).replace("\n", "")
            champion_search = difflib.get_close_matches(champion_name, champions_array)
            player.champion = champion_search[0] if len(champion_search) > 0 else 'unknown'


        gold_image = playerRow[0:int(playerRow.shape[0] * 0.55), (x_gold - 5):(x_gold + w_gold + 5)]
        gold_image = preprocessImg(gold_image)
        player.gold = readGoldAndDmg(gold_image).replace("\n", "")

        dmg_image = playerRow[0:int(playerRow.shape[0] * 0.55), (x_dmg - 5):(x_dmg + w_dmg + 5)]
        dmg_image = preprocessImg(dmg_image)
        player.dmg = readGoldAndDmg(dmg_image).replace("\n", "")


        #kda was not reading as good as the other ones, in this case we have divided into digits and read individually and if
        #it fails, we read the whole text
        kda_image = playerRow[0:int(playerRow.shape[0] * 0.55), (x_kda - 5):(x_kda + w_kda + 5)]
        kda_image = preprocessImg(kda_image)
        cv2.imwrite(f"./output/Item_{i}_kda.png", kda_image)

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
            # img_digit = cv2.copyMakeBorder(img_digit, 5,5,5,5, borderType=cv2.BORDER_CONSTANT, value=255)
            img_digit = preprocessImg2(img_digit)
            cv2.imwrite(f"./output/Item_{i}_digit.png", img_digit)
            character = readSingleDigit(img_digit)
            kda_string += character[0] if character != '' and character is not None else '/'
        
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

        player.kda = kda_string
        players.append(player.toJson())
        
        i += 1
        i %= 5
    
    print(players)

        

