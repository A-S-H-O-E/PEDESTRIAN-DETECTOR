import cv2
bodyclassifier = cv2.CascadeClassifier('PRO-C106-ProjectSolution-main/haarcascade_fullbody.xml')
vid = cv2.VideoCapture('PRO-C106-ProjectSolution-main/walking.avi')
while True:
    ret,frame = vid.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = bodyclassifier.detectMultiScale(grey,1.2,3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,150,60),2)
        cv2.imshow('pedstrians',frame)
    if(cv2.waitKey(1) == 32):
        break
vid.release()
cv2.destroyAllWindows()