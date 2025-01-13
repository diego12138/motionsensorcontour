import cv2
import streamlit as st
import numpy as np

st.title('Motion Sensor')
run = st.checkbox('Start Sensing')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
bg_sub = cv2.createBackgroundSubtractorMOG2()   

while run:
    ret, frame = cap.read()
    
    fgmask = bg_sub.apply(frame)
    
    contours, _=cv2.findContours(fgmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame, channels='BGR')
    
cap.release()
cv2.destroyAllWindows()