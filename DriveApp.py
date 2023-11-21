import csv

import numpy as np
import cv2
from keras.models import load_model
import imageio
from keras.models import Model
model = load_model('Autopilot.h5')
from skimage.transform import resize


def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))

    steering_angle = steering_angle * 100
    return steering_angle


def keras_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


out = open("./data/Night.csv", 'a', newline='')
csv_writer = csv.writer(out)
steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

# while (cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
#     steering_angle = keras_predict(model, gray)
#     print(steering_angle)
#     cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
#     smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
#         steering_angle - smoothed_angle) / abs(
#         steering_angle - smoothed_angle)
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
#     dst = cv2.warpAffine(steer, M, (cols, rows))
#     cv2.imshow("steering wheel", dst)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
i = 1
while (cv2.waitKey(10) != ord('q')):
    frame = imageio.imread("../../dataset/night/night_" + str(i) + ".jpg", pilmode="RGB")
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    steering_angle = keras_predict(model, gray)

    activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    image = keras_process_image(gray)
    vec = activation_model.predict(image)
    outputDict = []
    outputDict.append("night_" + str(i))
    outputDict.append(str(steering_angle))
    outputDict.append(vec)
    csv_writer.writerow(outputDict)

    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
            steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    i = i + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.close()
cv2.destroyAllWindows()
