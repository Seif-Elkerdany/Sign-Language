#modules
import os
import mediapipe as mp
import cv2
import csv
import math as m

#objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

Data_Dir = r"D:\Projects\Sign language\My image data"

#iterate through files
#dir_ is the label
for dir_ in os.listdir(Data_Dir):
    for img_path in os.listdir(os.path.join(Data_Dir,dir_)):
        #if img_path.endswith("jpeg"or"jpg"or"JPG"or"JPEG"):
            #read every image
            img = cv2.imread(os.path.join(Data_Dir,dir_,img_path))
            #conv every image from BGR into RGB
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #process every image
            results = hands.process(img_RGB)
    
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                Data_Xs=[]
                Data_Ys=[]
                
                for landmark in landmarks.landmark:
                    #transform points
                    landmark.x -= landmarks.landmark[0].x
                    landmark.y -= landmarks.landmark[0].y
                
                #Distance
                distance = m.sqrt(pow((landmarks.landmark[17].x - landmarks.landmark[5].x),2) + pow((landmarks.landmark[17].y - landmarks.landmark[5].y),2))
                
                for landmark in landmarks.landmark:
                    #Normalization
                    x = landmark.x / distance
                    y = landmark.y / distance
                    
                    Data_Xs.append(x)
                    Data_Ys.append(y)

                image_Data=[]
                image_Data.extend(Data_Xs)
                image_Data.extend(Data_Ys)
                image_Data.append(dir_)

                with open("MyTraining_Data.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(image_Data)
