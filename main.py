import cv2

face_ref = cv2.CascadeClassifier("face.xml")

camera = cv2.VideoCapture(0)

def face_detect(frame):
  optimized = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  faces = face_ref.detectMultiScale(optimized, scaleFactor=1.5, minSize=(900, 900), minNeighbors=6)
  return faces

def box(frame):
  for x, y, w, h in face_detect(frame):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

def close_window():
  camera.release()
  cv2.destroyAllWindows()
  exit()

def main():
  while True:
    _, frame = camera.read()
    drawer = box(frame)
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      close_window()

if __name__ == '__main__':
  main()
