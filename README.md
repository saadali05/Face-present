# Face Recognition Attendance System

This project implements a face recognition-based attendance system using Flask, OpenCV, and a K-Nearest Neighbors classifier. The system captures images using a webcam, detects faces, identifies them using a pre-trained model, and records attendance.

## Features

- **Face Detection and Recognition**: Uses OpenCV's Haar Cascades for face detection and a K-Nearest Neighbors classifier for face recognition.
- **Attendance Logging**: Records attendance in CSV files.
- **User Management**: Allows adding new users with their face data and updating or deleting existing users.
- **Web Interface**: Provides a simple web interface to view and manage attendance records.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/face-recognition-attendance.git
    cd face-recognition-attendance
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download Haar Cascade for face detection**:
    Download the `haarcascade_frontalface_default.xml` file from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the project directory.


## Usage

1. **Run the Flask application**:
    ```sh
    python app.py
    ```

2. **Access the web interface**:
    Open a web browser and go to `http://127.0.0.1:5000/`.

3. **Home Page**:
    - View the list of attendees and their check-in times.
    - See the total number of registered users.

4. **Add New User**:
    - Navigate to `http://127.0.0.1:5000/add` to add a new user.
    - Enter the username and user ID, and capture the required number of face images.

5. **Start Attendance**:
    - Navigate to `http://127.0.0.1:5000/start` to start the face recognition attendance process.
    - The system will use the webcam to capture images, detect faces, and log attendance for recognized users.

6. **Manage Users**:
    - Navigate to `http://127.0.0.1:5000/view_edit` to view, update, or delete users.

## Functions

### totalreg()
Returns the total number of registered users.

### extract_faces(img)
Extracts face regions from an image.

### identify_face(facearray)
Identifies a face using the pre-trained KNN model.

### train_model()
Trains the KNN model on the registered users' face data.

### extract_attendance()
Extracts the attendance records from the CSV file.

### add_attendance(name)
Adds a user's attendance record to the CSV file.

### getallusers()
Retrieves the list of all registered users.

## Future Work

- **Login Page**: Implement a login system to secure access to the attendance system and user management features.
- **Integration with Educational Systems**: Collaborate with educational institutions to integrate the attendance system with existing student management platforms.
- **Integration with Google Sheets**: Allow seamless synchronization of attendance records with Google Sheets for easy access and analysis.
- **Model Stacking for Performance Improvement**: Experiment with ensemble learning techniques like model stacking to improve the accuracy and robustness of the face recognition model.
- **Cloud Storage**: Implement cloud storage solutions such as AWS S3 or Google Cloud Storage to store attendance records securely and access them from anywhere.
- **Mobile Integration**: Develop a mobile application companion to the web interface for convenient access to attendance records and management features on smartphones and tablets.
- **Real-time Notification and Alerts**: Implement real-time notifications and alerts for administrators and users to receive updates on attendance status, late arrivals, or any irregularities detected by the system.

By incorporating these enhancements, the face recognition attendance system can become more versatile, user-friendly, and seamlessly integrated into educational workflows. Contributions and feedback on these future developments are highly encouraged and welcomed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
