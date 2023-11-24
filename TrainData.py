import csv
import cv2
from keras.models import load_model
import numpy as np
from keras.models import Model
import imageio
from TrainModel import loadFromPickle, augmentData

features_directory = './data/'
labels_file = './data/driving_log.csv'


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


def load_autopilot_model():
    # 加载预训练的 Autopilot 模型
    model = load_model('Autopilot.h5')
    return model


def get_layer_output(model, layer_number, input_data):
    # 获取指定层的输出内容
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[layer_number].output)
    layer_output = intermediate_layer_model.predict(input_data)
    return layer_output


def data_loading():
    logs = []
    # 一个list存入feature文件
    paths = []
    labels = []
    with open(labels_file, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            logs.append(line)
        log_labels = logs.pop(0)
    for i in range(len(logs)):
        img_path = logs[i][0]
        print(img_path)
        img_path = features_directory + 'IMG' + (img_path.split('IMG')[1]).strip()
        paths.append(img_path)
        labels.append(logs[i][3])
    return paths, labels


def main():
    i = 0
    out = open("./data/Origin.csv", 'a', newline='')
    csv_writer = csv.writer(out)

    model = load_model("Autopilot.h5")
    paths, labels = data_loading()

    steer = cv2.imread('steering_wheel_image.jpg', 0)
    rows, cols = steer.shape
    smoothed_angle = 0

    while i < len(paths):
        img_path = paths[i]
        frame = imageio.imread(img_path, pilmode="RGB")
        gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
        steering_angle = keras_predict(model, gray)

        activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)

        image = keras_process_image(gray)
        vec = activation_model.predict(image)

        outputDict = [
            "Origin_" + str(i),
            np.array2string(vec, separator=', ', threshold=np.inf),
            str(labels[i]),
            str(steering_angle),
        ]

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


if __name__ == "__main__":
    main()
