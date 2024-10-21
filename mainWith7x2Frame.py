import cv2
from util import get_parking_spots_bboxes, empty_or_not

mask = './mask_crop.png'
video_path = './data/parking_crop_loop.mp4' 

mask = cv2.imread(mask, 0)
# mask.png and video.mp4 is already in our data set and our project directory. 

cap = cv2.VideoCapture(video_path)
# we read our mask png and turned it to a numpy array

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

# print(spots[0])
# empty coordinates for first run

step = 30
spots_status = [None for j in spots]

frame_nmr = 0
ret = True
while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1+h, x1:x1+w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr = frame_nmr + 1

cap.release()
cv2.destroyAllWindows()
