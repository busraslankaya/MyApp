from flask import Flask, request, jsonify, make_response
from numpy.lib.type_check import imag
import cv2
import mediapipe as mp
import base64
import numpy as np


app = Flask(__name__)
app.config["DEBUG"] = False
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

class handGesture():
    #Ititializes a Mediapipe hand object
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_con=0.5, min_tracking_con=0.5):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detectionCon = min_detection_con
        self.trackCon = min_tracking_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def drawHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
     
        if self.results.multi_hand_landmarks:
            for each_handLms in self.results.multi_hand_landmarks:
                if draw: #red point and green lines
                    self.mpDraw.draw_landmarks(img, each_handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def positions(self, img, handNo=0):
        #landmarks numbers list
        lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lms in enumerate(myHand.landmark):
                h, w, c = img.shape #height, weight, channels
                cx, cy = int(lms.x * w), int(lms.y * h)
                lmsList.append([id, cx, cy])

        return lmsList

tipIds = [4, 8, 12, 16, 20]
detection = handGesture(min_detection_con=0.75)

def get_fingers(img):

    while True:
        success, img = cap.read()
        img = detection.drawHands(img)
        lmsList = detection.positions(img)

        if len(lmsList) != 0:
            fingers = []
            if( lmsList[tipIds[1]][1] > lmsList[tipIds[4]][1]):
                # Right Thumb
                if lmsList[tipIds[0]][1] > lmsList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Right 4 Fingers
                for id in range(1, 5):
                    if lmsList[tipIds[id]][2] < lmsList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                fingerNum = fingers.count(1)

            if( lmsList[tipIds[1]][1] < lmsList[tipIds[4]][1]):
                # Left Thumb
                if lmsList[tipIds[0]][1] < lmsList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Left 4 Fingers
                for id in range(1, 5):
                    if lmsList[tipIds[id]][2] < lmsList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                fingerNum = fingers.count(1)
        

            cv2.putText(img, str(fingerNum), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        8, (0, 0, 0), 5)

            if fingerNum==5:
                cv2.putText(img, "stop", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2, (0, 0, 0), 4)
                print(fingerNum , "stop")
            if fingerNum==1:    
                cv2.putText(img, "forward", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2, (0, 0, 0), 4)
                print(fingerNum , "forward")
            if fingerNum==2:
                cv2.putText(img, "backward", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2, (0, 0, 0), 4)
                print(fingerNum , "backward")
            if fingerNum==3:
                cv2.putText(img, "right", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2, (0, 0, 0), 4)  
                print(fingerNum , "right") 
            if fingerNum==4:
                cv2.putText(img, "left", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2, (0, 0, 0), 4) 
                print(fingerNum , "left")   
            return fingerNum[img]
        cv2.imshow("Image", img)
        cv2.waitKey(1)


def b64_to_img(img):
    encoded_string = base64.b64encode(img)
    decoded_string = base64.b64decode(encoded_string)
    base64_img = np.fromstring(decoded_string, dtype=np.uint8)
    base64_img = base64_img.reshape(img.shape)

@app.route('/api/fingercounter', methods=['GET'])
def trace():
    
    try:
        values = request.get_data()
        base64_img = values["img"]
        image = b64_to_img[base64_img]
        finger_count=get_fingers(image)
        resp = jsonify(message='success', finger_count=finger_count)
        return make_response(resp, 200)

    except Exception as ex:
        resp = jsonify(message=str(ex.args[0]), finger_count=-1)
        return make_response(resp, 500)

if __name__ == '__main__':
    app.run()
    
