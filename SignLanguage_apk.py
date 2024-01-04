import cv2
import mediapipe as mp
import numpy as np
import math as m
import pickle

cap = cv2.VideoCapture(0)

model_dict = pickle.load(open(r"C:\Users\Saif Elkerdany\Model_mydata.pkl", 'rb'))
model = model_dict

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7)

while True:
  fps , frame = cap.read()
  
  H, W, _ = frame.shape
  
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  #processing the images
  results = hands.process(frame_rgb)
  if results.multi_hand_landmarks:
    landmarks = results.multi_hand_landmarks[0]
    
    x_ = []
    y_ = []
          
    X_ = []     
    Y_ = [] 
    #draw
    for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
            
    #collecting data from video
    for landmark in landmarks.landmark:
      #translation
      landmark.x -= landmarks.landmark[0].x
      landmark.y -= landmarks.landmark[0].y
      
    distance = m.sqrt(pow((landmarks.landmark[17].x - landmarks.landmark[5].x),2) + pow((landmarks.landmark[17].y - landmarks.landmark[5].y),2))
    
    for landmark in landmarks.landmark:
      x = landmark.x  / distance
      y = landmark.y / distance
      x_.append(x)
      y_.append(y)
    
    VideoData=[]
    VideoData.extend(x_)
    VideoData.extend(y_)
    
    prediction = model.predict([VideoData])
    predicted_char = prediction[0]
    print(predicted_char)
    
    frame = cv2.putText(frame, predicted_char, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (250,0,255), 3,cv2.LINE_AA)
    
  cv2.imshow('frame', frame)
  cv2.waitKey(1)
