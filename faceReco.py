import cv2
import os
import sys
import numpy as np
import math
import face_recognition
import pickle


def face(face_distance, face_match_threshold=0.6):
    range_val = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return f"{round(linear_val * 100, 2)}%"
    else:
        value = (
            linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
        ) * 100
        return f"{round(value, 2)}%"


class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True

    def encode_faces(self):
        if os.path.exists("FursanCompany.pkl"):
            with open("FursanCompany.pkl", "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print("Loaded known faces.")
        else:
            for image in os.listdir("face"):
                face_image = face_recognition.load_image_file(f"face/{image}")
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)

            with open("FursanCompany.pkl", "wb") as f:
                pickle.dump((self.known_face_encodings, self.known_face_names), f)
            print("Saved known faces to file.")

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Video source is not found....")

        while True:
            ret, frame = video_capture.read()

            # Kamera Setting Mirror / engga
            frame = cv2.flip(frame, 1)

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                rgb_small_frame = rgb_small_frame.astype(np.uint8) 
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations
                )

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding
                    )
                    name = "unknown"
                    confidence = "unknown"

                    face_distance = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distance)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face(face_distance[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            # display anotasi
            for (top, right, bottom, left), name in zip(
                self.face_locations, self.face_names
            ):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font size
                    (255, 255, 255),
                    1,
                )

                if name != 'unknown':
                    print("Detected: ", name)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.encode_faces()
    fr.run_recognition()
