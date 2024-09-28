import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread("ada.jpg")

def face_detection(frame):
    cv2.rectangle(frame, (20, 20), (200, 80), (255, 0, 0), 2)
    cv2.putText(frame, "Face Detection", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Face Detection", frame)
    cv2.waitKey(1)  
    
    face_detection(frame)  
  

def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, img)
        if result["verified"]:
            face_match = True
            print("Face Match Detected!")
            
        else:
            face_match = False
    except Exception as e:
        print(f"Error: {str(e)}")

while True:
    ret, img = cap.read()
    if ret:
        if counter % 30 == 0:  # Run the face detection check every 30 frames
            try:
                threading.Thread(target=check_face, args=(img.copy(),)).start()
            except ValueError:
                pass
        
        counter += 1
        
        if face_match:
            cv2.putText(img, "Face Match Detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Match!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        
        cv2.imshow("Face Detection", img)
        
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release resources after exiting the loop
cap.release()
cv2.destroyAllWindows()
