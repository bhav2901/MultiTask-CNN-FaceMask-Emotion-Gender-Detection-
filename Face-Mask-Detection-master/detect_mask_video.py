# import necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import argparse
import cv2
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Function to detect faces, predict masks, classify gender, and detect smiles
def detect_mask_gender_and_smile(frame, faceNet, maskNet, genderNet, smileNet, classes):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    mask_preds = []
    gender_preds = []
    smile_preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box stays within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]

            if face.size == 0:  # Check if face is empty
                continue

            # Preprocess face for gender classification and smile detection
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (96, 96))
            face_normalized = face_resized.astype("float") / 255.0
            face_input = img_to_array(face_normalized)
            face_input = np.expand_dims(face_input, axis=0)

            # Predict mask
            face_mask = cv2.resize(face_rgb, (224, 224))
            face_mask = img_to_array(face_mask)
            face_mask = preprocess_input(face_mask)
            mask_preds = maskNet.predict(np.array([face_mask]))

            # Predict gender
            gender_conf = genderNet.predict(face_input)[0]
            gender_idx = np.argmax(gender_conf)
            gender_label = classes[gender_idx]
            gender_label = "{}: {:.2f}%".format(classes[gender_idx], gender_conf[gender_idx] * 100)
            gender_preds.append(gender_label)

            # Predict smile
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_smile = cv2.resize(face_gray, (28, 28))
            face_smile = face_smile.astype('float') / 255.0
            face_smile = img_to_array(face_smile)
            face_smile = np.expand_dims(face_smile, axis=0)
            smile_preds = smileNet.predict(face_smile)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    return (locs, mask_preds, gender_preds, smile_preds)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required=True, 
    help='path to where the face cascade resides')
ap.add_argument('-m', '--model', required=True, 
    help='path to the pre-trained smile detector CNN')
ap.add_argument('-v', '--video', 
    help='path to the (optional) video file')
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args['cascade'])
maskNet = load_model("mask_detector.model")
genderNet = load_model('gender_detection.model')
smileNet = load_model(args["model"])
classes = ['man', 'woman']

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# if a video path was not supplied, grab the reference to the webcam
if not args.get('video', False):
    print('[INFO] starting video capture...')
    vs = VideoStream(src=0).start()

# otherwise, load the video
else:
    vs = cv2.VideoCapture(args['video'])

# keep looping
while True:
    if args.get('video', False):
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
    else:
        frame = vs.read()
        frame = frame if args.get('video', False) is None else frame[1]

    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        # extract the face ROI, resize it, preprocess it, and prepare it for classification
        face = frame[y:y+h, x:x+w]
        (locs, mask_preds, gender_preds, smile_preds) = detect_mask_gender_and_smile(face, faceNet, maskNet, genderNet, smileNet, classes)

        if len(locs) > 0:
            (startX, startY, endX, endY) = locs[0]
            (mask, withoutMask) = mask_preds[0]

            # Determine label and color for face mask detection
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw bounding box, label, and gender classification on the frame
            cv2.rectangle(frameClone, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            cv2.putText(frameClone, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.putText(frameClone, gender_preds[0], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Display smile detection result
            smile_label = "Not Smiling" if smile_preds[0][0] > 0.5 else "Smiling"
            cv2.putText(frameClone, smile_label, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    # display the output frame
    cv2.imshow("Face Mask, Gender, and Smile Detection", frameClone)

    # break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
if not args.get('video', False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
