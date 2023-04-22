import cv2


def objectTracker2(facePath, eyePath):
    faceCascade = cv2.CascadeClassifier(facePath)
    eyeCascade = cv2.CascadeClassifier(eyePath)
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    while True:  # try to get the first frame
        rval, frame = vc.read()
        # Capture frame-by-frame
        ret, frame = vc.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                             minNeighbors=1, minSize=(40, 40),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # select face as region of interest
            roi_g = gray[y:y + h, x:x + h]
            roi_c = frame[y:y + h, x:x + h]
            # within region of interest find eyes
            eyes = eyeCascade.detectMultiScale(roi_g)
            # for each eye
            for (ex, ey, ew, eh) in eyes:
                # draw retangle around eye
                cv2.rectangle(roi_c, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")