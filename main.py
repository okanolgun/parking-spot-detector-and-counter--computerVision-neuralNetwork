import cv2
from util import get_parking_spots_bboxes

mask = './mask_crop.png'
video_path = './data/parking_crop_loop.mp4'

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

print(spots[0])

ret = True
while ret:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()