from pathlib import Path
from pickle import dump, load

import numpy as np

import face_recognition
from face_recognition.api import _raw_face_landmarks, face_detector


def scan_all_faces_aux(directory: Path):
    faces = []
    for f_idx, photo in enumerate(directory.glob('*.jpg')):
        image = face_recognition.load_image_file(photo)
        # rrr = face_detector.run(image)
        # print(str(photo), rrr)
        locations = face_recognition.face_locations(image)
        landmarks = [i.parts() for i in _raw_face_landmarks(image, locations)]
        encodings = face_recognition.face_encodings(image, locations, num_jitters=2, model='large')
        for location, landmark, encoding in zip(locations, landmarks, encodings):
            index = len(faces)
            faces.append(dict(
                photo=photo,
                location=location,
                landmarks=landmark,
                encoding=encoding,
                index=index
            ))
            if len(faces) % 10 == 0:
                print(f'Scanned {len(faces)} faces from {f_idx} photos')
    return faces

def scan_all_faces(directory: Path):
    try:
        with (directory / 'faces.pkl').open('rb') as f:
            faces = load(f)
            print(f'Loaded {len(faces)} faces from file')
    except Exception as e:
        faces = scan_all_faces_aux(directory)
        dump(faces, (directory / 'faces.pkl').open('wb'))

    encodings = np.array([face['encoding'] for face in faces])
    return encodings, faces