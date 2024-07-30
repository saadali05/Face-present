import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from shutil import rmtree
import time

app = Flask(__name__)

nimgs = 5

imgBackground = cv2.imread("bg.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    
    try:
        if len(faces) == 0:
            raise ValueError("No faces found database.")
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
    except ValueError as e:
        print("Error:", e)
        
    time.sleep(5)  # Added delay after training

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    
    time.sleep(5)  # Added delay after training


def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Handle the case where the camera is not opened successfully
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='Error: Unable to open camera.')

    while ret:
        ret, frame = cap.read()
        if frame is None:
            # Handle the case where the frame is None (e.g., camera disconnected)
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                                   datetoday2=datetoday2,
                                   mess='Error: Unable to capture frame.')
        # Log the dimensions of the captured frame
        print("Frame dimensions:", frame.shape)

        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            # Handle the case where the frame dimensions are invalid
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                                   datetoday2=datetoday2,
                                   mess='Error: Invalid frame dimensions.')

        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, f'{identified_person}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            # Resize the frame to match the dimensions of the background image
            frame = cv2.resize(frame, (640, 480))
            try:
                imgBackground[162:162 + frame.shape[0], 55:55 + frame.shape[1]] = frame
            except ValueError:
                # Handle the case where the shapes are not compatible
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                                       datetoday2=datetoday2,
                                       mess='Error: Frame dimensions are not compatible with the background image.')
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),datetoday2=datetoday2) 
    
    time.sleep(5)  # Added delay after training


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        else:
            return render_template('home.html', mess='User already exists. Please use a different ID.',
                                   names=[], rolls=[], times=[], l=0, totalreg=totalreg(), datetoday2=datetoday2)

        i = 0
        cap = cv2.VideoCapture(0)
        while i < nimgs:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1

            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27 or i >= nimgs:
                break

        cap.release()
        cv2.destroyAllWindows()

        print('Training Model')
        train_model()  # Ensure model is trained after adding new user

        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2)
    else:
        return redirect(url_for('home'))  # If not a POST request, redirect to the home page





@app.route('/delete/<roll>', methods=['GET'])
def delete_user(roll):
    userlist, names, rolls, l = getallusers()
    user_index = rolls.index(roll)
    user_folder = userlist[user_index]
    # Check if the directory is empty
    if not os.listdir(f'static/faces/{user_folder}'):
        # Directory is empty, delete it
        os.rmdir(f'static/faces/{user_folder}')
    else:
        # Directory is not empty, remove all files and then delete it
        rmtree(f'static/faces/{user_folder}')
    train_model()
    return redirect(url_for('view_edit'))


@app.route('/update_user', methods=['POST'])
def update_user():
    old_id = request.form['old_id']
    new_name = request.form['new_name']
    new_id = request.form['new_id']

    userlist, names, rolls, l = getallusers()
    user_index = rolls.index(old_id)
    user_folder = userlist[user_index]

    # Rename the user folder with the new name and ID
    os.rename(f'static/faces/{user_folder}', f'static/faces/{new_name}_{new_id}')

    # Retrain the model
    train_model()
    return '', 200



@app.route('/view_edit')
def view_edit():
    userlist, names, rolls, l = getallusers()
    return render_template('view_edit.html', userlist=userlist, names=names, rolls=rolls, l=l)


@app.route('/attendance_data', methods=['GET'])
def attendance_data():
    names, rolls, times, l = extract_attendance()
    return {
        'names': names.tolist(),
        'rolls': rolls.tolist(),
        'times': times.tolist(),
        'totalreg': totalreg()
    }


if __name__ == '__main__':
    app.run(debug=True)
