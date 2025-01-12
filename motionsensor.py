import cv2

cap = cv2.VideoCapture(0)
bg_sub = cv2.createBackgroundSubtractorMOG2()   

while True:
    ret, frame = cap.read()
    
    fgmask = bg_sub.apply(frame)
    
    contours, _=cv2.findContours(fgmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()