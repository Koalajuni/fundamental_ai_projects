import dlib
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, detectorPath, predictorpath):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictorpath)
        self.image = None
        self.detections = None

    def load_image(self, image):
        self.image = image

    def detect_faces(self, upsample_num_times=1):
        if self.image is None:
            raise ValueError("Image not loaded.")
        self.detections = self.detector(self.image, upsample_num_times)
        return self.detections

    def draw_faces(self):
        if self.detections is None:
            raise ValueError("No faces.")
        for det in self.detections:
            x, y, w, h = det.left(), det.top(), det.width(), det.height()
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def draw_landmark(self):
        if self.detections is None:
            raise ValueError("No faces.")
        objects = dlib.full_object_detections()

        for det in self.detections:
            shape = self.predictor(self.image, det)
            objects.append(shape)

            for point in shape.parts():
                # print(point)
                cv2.circle(self.image, center=(point.x, point.y),
                           radius=6, color=(0, 0, 255))
