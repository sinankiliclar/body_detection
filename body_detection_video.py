import cv2 

cap=cv2.VideoCapture("D:\opencv_udemy/08_body_detection/body.mp4")
body_cascade=cv2.CascadeClassifier("D:\opencv_udemy/08_body_detection/fullbody.xml")


while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    bodies=body_cascade.detectMultiScale(gray,1.3,4)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(y+h,x+w),(0,0,255),2)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1)==27:
        break


cap.release()
cv2.destroyAllWindows()