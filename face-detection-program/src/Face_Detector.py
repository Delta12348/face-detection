import cv2 #open cv library
from random import randrange #used to give the rectangle a random color

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #Feeding the AI all the data required, in this case the HaarCascade algorithm Imported from open CV repository and

analysis_type = input("Select type of analysis, 1 is image, 2 is video, default is webcam: ") #Asks the user what type of file requires analysis

def inputAnalysis(frame): #Function that takes a processed frame as input, can be video or image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # The cascade algorithm only analyzes brightness in pictures, transforming the frame of the video into gray optimzes greatly, especially in videos

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)#Process the greyframe with the cascade data
    for (x,y,w,h) in face_coordinates: #Asks openCV for the coordinates of the faces and its width and height 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256),randrange(128, 256),randrange(128, 256)), 2) #Since the coordinates of the face are already known, we can draw the rectangle in the specified rectangles in the original image
    return frame #Returns the original frame of the input with a rectangle drawn

def videoSetup(video): #Function that takes the parameter video
    webcam = cv2.VideoCapture(video) #This the file given by user
    while True: #While it keeps renderizing frames
        succesful_frame_read, frame = webcam.read() #Checks if frames are being renderized, if so reads them

        processedframe = inputAnalysis(frame=frame) #Calls and stores the result of inputAnalysis function in a variable

        cv2.imshow("Face Detection Program", processedframe) #Show the result of the analysis in a program called Face Detection Program
        key = cv2.waitKey(1) #This will wait for a key to be pressed to continue, if no input was registered in a milisecond continue with next frame
        if key == 81 or key == 113: #If key pressed is Q or q, exit the program
            break
    webcam.release() #This is used to turn of the webcam or video

def imageSetup(image): #Function that takes as a parameter an image
    frame = cv2.imread(image) #Takes the image of the user as a frame
    processedframe = inputAnalysis(frame=frame) #Calls and stores the result of inputAnalysis in a variable

    cv2.imshow("Face Detection Program", processedframe) #Shows the result in a program called Face Detection Program
    cv2.waitKey() #Waits for a key to end the program

if analysis_type == "1": 
    filename = input("Write name of file in this directory with relative path: ") #A little unefficient but funcitional. Asks the name of the image to analyze
    imageSetup(image=filename) #Calls the imageSetup function

elif analysis_type == "2": 
    filename = input("Write name of file in this directory with relative path: ")
    videoSetup(video=filename)

else:
    videoSetup(video=0) #To call the webcam in openCV you input 0
       