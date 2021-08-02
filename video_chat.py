import threading
from socket import *
import cv2
import struct
import zlib
import pickle
import time
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np
import csv
import pandas as pd

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
result_path = 'D:\github\result.csv'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry" ,"disgust", "scared",
            "happy", "sad", "surprised", "neutral"]

class Video_Server(threading.Thread):
    def __init__(self, port, version) :
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.ADDR = ('', port)
        if version == 4:
            self.sock = socket(AF_INET ,SOCK_STREAM)
        else:
            self.sock = socket(AF_INET6 ,SOCK_STREAM)
    def __del__(self):
        self.sock.close()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    def run(self):
        print("VIDEO server starts...")
        self.sock.bind(self.ADDR)
        self.sock.listen(1)
        conn, addr = self.sock.accept()
        print("remote VIDEO client success connected...")
        data = "".encode("utf-8")
        payload_size = struct.calcsize("L")		# 结果为4
        cv2.namedWindow('Remote', cv2.WINDOW_NORMAL)
        while True:
            while len(data) < payload_size:
                data += conn.recv(81920)
            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_size)[0]
            while len(data) < msg_size:
                data += conn.recv(81920)
            zframe_data = data[:msg_size]
            data = data[msg_size:]
            frame_data = zlib.decompress(zframe_data)
            frame = pickle.loads(frame_data)
            cv2.imshow('Remote', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break


class Video_Client(threading.Thread):
    def __init__(self ,ip, port, level, version):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.ADDR = (ip, port)
        if level <= 3:
            self.interval = level
        else:
            self.interval = 3
        self.fx = 1 / (self.interval + 1)
        if self.fx < 0.3:	# 限制最大帧间隔为3帧
            self.fx = 0.3
        if version == 4:
            self.sock = socket(AF_INET, SOCK_STREAM)
        else:
            self.sock = socket(AF_INET6, SOCK_STREAM)
        self.cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)
    def __del__(self) :
        self.sock.close()
        self.cap.release()
    def run(self):
        print("VIDEO client starts...")
        while True:
            try:
                self.sock.connect(self.ADDR)
                break
            except:
                time.sleep(3)
                continue
        print("VIDEO client connected...")
        results = {"angry":0, "disgust":0, "scared":0,
        "happy":0, "sad":0, "surprised":0, "neutral":0}
        while self.cap.isOpened():
            emotion_dict = {"angry":0, "disgust":0, "scared":0,
            "happy":0, "sad":0, "surprised":0, "neutral":0}
            for i in range(15):
                frame = self.cap.read()[1]
                # sframe = cv2.resize(frame, (0,0), fx=self.fx, fy=self.fx)
                # srame = imutils.resize(frame, width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

                # canvas = np.zeros((250, 300, 3), dtype="uint8")
                # frameClone = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True,
                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                    # the ROI for classification via the CNN
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = emotion_classifier.predict(roi)[0]
                    # emotion_probability = np.max(preds)
                    label = EMOTIONS[preds.argmax()]
                    emotion_dict[label] = emotion_dict[label] + 1
                    img = cv2.imread('emojis/' + label + '.png')
                else:
                    continue
                data = pickle.dumps(img)
                zdata = zlib.compress(data, zlib.Z_BEST_COMPRESSION)
                try:
                    self.sock.sendall(struct.pack("L", len(zdata)) + zdata)
                except:
                    break
                for i in range(self.interval):
                    self.cap.read()
            best_emotion = max(emotion_dict, key=emotion_dict.get)
            print(best_emotion)
            results[best_emotion] += 1
            print(results)
