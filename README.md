
# Hand Tracking and Angle Calculation

This project demonstrates real-time hand tracking using MediaPipe and OpenCV to calculate and display angles between hand joints.

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

### Step 1: Clone the Repository

```sh
git clone https://github.com/yourusername/hand-tracking.git
cd hand-tracking
```

### Step 2: Create and Activate a Virtual Environment

Create a virtual environment to manage your dependencies.

#### On Windows:

```sh
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```sh
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Use the provided `requirements.txt` file to install all the necessary dependencies.

```sh
pip install -r requirements.txt
```

### Step 4: Run the Script

After installing the dependencies, you can run the script to start hand tracking.

```sh
python hand_tracking.py
```

## Usage

1. Ensure your webcam is connected.
2. Run the script using the command above.
3. The script will open two windows: 
    - **Hand Tracking**: Displays the webcam feed with hand landmarks drawn.
    - **Hand Angles**: Displays a grid showing the calculated angles of various hand joints.

Press `ESC` to close the application.

## Requirements File

Create a `requirements.txt` file with the following content:

```txt
opencv-python
mediapipe
numpy
```

## Code Overview

- **calculate_angle**: Calculates the angle between three points.
- **create_grid_image**: Creates an image displaying the calculated angles.
- **main script**: Captures video input, processes each frame to detect hand landmarks, calculates angles, and displays the results.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)

Feel free to contribute to this project by opening issues or submitting pull requests.
