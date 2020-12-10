# Step1 : Find faces in Our Image (Haar Algorithm)
# Step2 : Find Smile Within the Faces( Haar Algorithm)
# Step3 : Label the Face if it's Smiling
import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  #cv2.data.haarcascades + is used to take out the error
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# Grab Webcam Feed

webcam = cv2.VideoCapture(0)
while True:
    #Read The Current Frame From the webcam
    successfull_frame_read , frame = webcam.read()

    if not successfull_frame_read: #if Error , Abort
        break

    # Change the Images to GrayScale  , As RGB has 2 Channels but Grayscale has 1 Channel so optimized
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # cvt -> Converting the frame from colour to Gray

    #Detect Faces first ( In Array of Points)
    faces = face_detector.detectMultiScale(frame_grayscale) #Multiscale use small or big facs too


    #Detect Smiles
    smiles = smile_detector.detectMultiScale(frame_grayscale,scaleFactor=1.7,minNeighbors=20) #ScaleFactor is to blur the Image so that it's easy (1.7 is perfect)
    # minNeighbours =
    #print(faces)  # List of List -> gives multiple points

    for(x,y,w,h) in faces: # w-> Width and h-> hight and y-> top point of the rec (Face Detection within each those Faces)

        #Draw a Rectangle around the Face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4) #Numbers are RGB , in openCV it's BGR


        # Get the Sub Frame (Using N-Dimensional array Slicing)
        the_face = frame[y:y+h,x:x+w] #OpenCV is build in Numpy so we can do this!  Actually it's Slicing



        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY) #Running Grayscale on the Face and NOT the Frame

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)


        # # Find all the Smile in the Face
        # for(x1,y1,w1,h1) in smiles:
        #     cv2.rectangle(the_face,(x1,y1),(x1+w1,y1+h1),(50,50,200),4)
        #     #draw all the rectangles around the smile




    # for(x,y,w,h) in smiles: # w-> Width and h-> hight and y-> top point of the rec (WE CAN USE SAME VARIABLES AS THEY NEVER OVERLAP, IF NESTED WE CAN'T)
    #
    #     #Draw a Rectangle around the Smile
    #     cv2.rectangle(the_face,(x,y),(x+w,y+h),(50,50,200),4) #Numbers are RGB , in openCV it's BGR


    if  len(smiles) > 0: #Writing Smiling # Greater than 0 ( i.e if More than 1 Smile)
        cv2.putText(frame, 'smiling',(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

    #Show the Current Frame
    cv2.imshow('serious', frame)

    #Display
    cv2.waitKey(1)        # 1 stands for 1 Millisecond -> so real Time

webcam.release()
cv2.destroyAllWindows()
#show Current Frame

