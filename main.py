import cv2
import numpy as np
import csv
from datetime import datetime
import face_recognition
from typing import Union
from enroll_n_save import enroll_face

# Load enrolled data from CSV
def load_enrolled_data() -> list:
    enrolled_data = []
    try:
        with open("enrolled_data.csv", "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                name = row[0]
                encoding = np.array(list(map(float, row[1:])))
                enrolled_data.append((name, encoding))
    except FileNotFoundError:
        pass
    return enrolled_data

# Function to mark attendance
def mark_attendance(name: str, students: list, csv_writer: csv.DictWriter, current_day: int, current_month: str) -> None:
    if name in known_faces_names and name in students and name not in attendance_recorded:
        students.remove(name)
        attendance_recorded.add(name)
        csv_writer.writerow({"ID": known_faces_names.index(name) + 1, "Full Name": name, f"{current_day}/{current_month}": 1})

# Initialize variables
known_faces_data = load_enrolled_data()
known_face_encodings = [data[1] for data in known_faces_data]
known_faces_names = [data[0] for data in known_faces_data]
students = known_faces_names.copy()
attendance_recorded = set()

# Function to mark attendance based on face recognition
def mark_attendance_ui() -> None:
    now = datetime.now()
    current_month: str = now.strftime("%B_%Y")
    current_day: int = now.day
    csv_filename: str = f"{current_month}_attendancerecord.csv"

    with open(csv_filename, mode="a", newline="") as csv_file:
        fieldnames = ["ID", "Full Name"] + [f"{day}/{current_month}" for day in range(1, 32)]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if csv_file.tell() == 0:
            csv_writer.writeheader()

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not open camera.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                if face_distance.any():  # Check if any element in the array is true
                    best_match_index = np.argmin(face_distance)
                    if matches[best_match_index]:
                        name = known_faces_names[best_match_index]
                        mark_attendance(name, students, csv_writer, current_day, current_month)

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (left * 4, top * 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
# Call the modified function
mark_attendance_ui()
