import cv2
import datetime
import dlib
import multiprocessing
import numpy as np
import time
import sys
import traceback

from timeout import set_timeout

def make_label(
    coordinates: tuple,
    img_path: str,
    img_resolution: tuple,
    location: str,
    brightness: str,
    eye_aspect_ratio: tuple,
    head_pose: tuple,
    facial_occupancy_ratio: str, # proxy for distance from camera
    glasses: bool,
    contacts: bool,
    timestamp: str
) -> str:
    """
    Makes a label string from the given parameters

    :param coordinates: coordinates of the point
    :param img_path: path to the image
    :param img_resolution: resolution of the image
    :param location: location at the time of capture
    :param brightness: brightness of the image
    :param eye_aspect_ratio: eye aspect ratio's of the user
    :param head_pose: head pose of the user
    :param facial_occupancy_ratio: ratio of the face in the frame
    :param glasses: whether the user is wearing glasses
    :param contacts: whether the user is wearing contacts
    :param timestamp: timestamp of the capture
    """
    head_pose_str = "|".join(";".join(str(x) for x in part) for part in head_pose)
    label = f"{';'.join(map(str, coordinates))}," \
            f"{img_path}," \
            f"{';'.join(map(str, img_resolution))}," \
            f"{location}," \
            f"{brightness}," \
            f"{';'.join(map(str, eye_aspect_ratio))}," \
            f"{head_pose_str}," \
            f"{facial_occupancy_ratio}," \
            f"{str(glasses)}," \
            f"{str(contacts)}," \
            f"{timestamp}"
    return label
    
def parse_label(label: str) -> dict:
    """
    Parses the label string into a dictionary

    :param label: label string
    :return: dictionary of the label
    """
    values = label.split(',')
    head_pose_parts = [tuple(part.split(';')) for part in values[6].split('|')]
    parsed_label = {
        "coordinates": tuple(map(int, values[0].split(';'))),
        "img_path": values[1],
        "img_resolution": tuple(map(int, values[2].split(';'))),
        "location": values[3],
        "brightness": float(values[4]),
        "eye_aspect_ratio": tuple(map(float, values[5].split(';'))),
        "head_pose": tuple(head_pose_parts),
        "facial_occupancy_ratio": float(values[7]),
        "glasses": values[8].lower() == 'true',
        "contacts": values[9].lower() == 'true',
        "timestamp": values[10]
    }
    return parsed_label

def save_current_progress(env_prog_path: str, current_point: tuple, last_label_i: int):
    """
    Saves the current progress to the environment progress file

    :param env_prog_path: path to the environment progress file
    :param current_point: current point in the environment
    :param last_label_i: last label index
    """
    with open(env_prog_path, 'w') as file:
        file.write(f"{current_point[0]},{current_point[1]},{last_label_i}\n")

def get_current_progress(env_prog_path: str) -> dict:
    """
    Parses the environment progress file and returns the current progress

    :param env_prog_path: path to the environment progress file
    :return: dictionary of the current progress
    """
    with open(env_prog_path, 'r') as file:
        lines = file.readlines()
        if len(lines) == 0:
            return {
                "current_point": (0, 0),
                "last_label_index": -1
            }
        last_line = lines[-1]
        last_point = tuple(map(int, last_line.split(',')))
        return (last_point[0], last_point[1]), last_point[2]

def classify_pose(angle, weak_threshold=5, strong_threshold=10):
    """
    Classifies the intensity of the pose based on the angle

    :param angle: angle to classify
    :param weak_threshold: weak threshold for the angle
    :param strong_threshold: strong threshold for the angle
    :return: classification of the angle
    """
    if abs(angle) < weak_threshold:
        return "center"
    elif weak_threshold <= abs(angle) < strong_threshold:
        return "weak"
    else:
        return "strong"
    
def calc_pitch(landmarks):
    """
    Computes the direction and intensity of the pitch of the users heads

    Uses a piecewise function to determine the intensity of the pitch

    If the ratio of the distance from the nose to midpoint of the brow is less than 0.42, the pitch is up
    If the ratio of the distance from the nose to midpoint of the brow is greater than 0.42, the pitch is down

    The pitch scales linearly between 0.3 and 0.42 and 0.42 and 0.6
    The top of the head is 15 degrees up and the bottom of the head is 15 degrees down

    The pitch is positive if the head is tilted up and negative if the head is tilted down

    Radians are implicit in the calculation, no need to convert to radians

    The calculation is a close approximation of the actual pitch

    :param landmarks: landmarks of the face
    :return: pitch of the head
    """
    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    left_eyebrow = (landmarks.part(17).x, landmarks.part(17).y)
    right_eyebrow = (landmarks.part(26).x, landmarks.part(26).y)
    eyebrow_midpoint = ((left_eyebrow[0] + right_eyebrow[0]) / 2, (left_eyebrow[1] + right_eyebrow[1]) / 2)
    chin = (landmarks.part(8).x, landmarks.part(8).y)

    nose_to_brow_dist = np.linalg.norm(np.array(nose_tip) - np.array(eyebrow_midpoint))
    nose_to_chin_dist = np.linalg.norm(np.array(nose_tip) - np.array(chin))

    ratio = nose_to_brow_dist / nose_to_chin_dist 
    if ratio < 0.42:
        pitch = (ratio - 0.42) * (15 / (0.42 - 0.3)) * -1
    else:
        pitch = (ratio - 0.42) * (15 / (0.6 - 0.42)) * -1

    return pitch

def calc_roll(landmarks):
    """
    Computes the direction and intensity of the roll of the users heads

    Uses the angle between the line connecting the corners of the eyes,
    as well as the angle between the line connecting the corners of the mouth,
    taking the average of the two angles to determine the roll

    Multiplied by -1 to invert the direction of the roll so left is positive and right is negative

    The roll is positive if the head is tilted to the left and negative if the head is tilted to the right

    :param landmarks: landmarks of the face
    :return: roll of the head
    """
    left_eye_corner = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye_corner = (landmarks.part(45).x, landmarks.part(45).y)
    left_mouth_corner = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth_corner = (landmarks.part(54).x, landmarks.part(54).y)

    eye_dx = right_eye_corner[0] - left_eye_corner[0]
    eye_dy = right_eye_corner[1] - left_eye_corner[1]
    mouth_dx = right_mouth_corner[0] - left_mouth_corner[0]
    mouth_dy = right_mouth_corner[1] - left_mouth_corner[1]

    eye_angle = np.arctan2(eye_dy, eye_dx) * 180 / np.pi * -1
    mouth_angle = np.arctan2(mouth_dy, mouth_dx) * 180 / np.pi * -1

    roll = (eye_angle + mouth_angle) / 2

    return roll

def calc_yaw(landmarks, roll):
    """
    Computes the direction and intensity of the yaw of the users heads

    Uses the angle between the line connecting the bridge of the nose and the tip of the nose

    90 degrees is subtracted from the angle to make the angle 0 when the head is facing forward

    The yaw is positive if the head is facing left and negative if the head is facing right

    The roll angle is added to the yaw angle to cancel it out

    :param landmarks: landmarks of the face
    :param roll: roll of the head
    :return: yaw of the head
    """
    node_bridge = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)]
    
    top = node_bridge[0]
    bottom = node_bridge[-1]
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]

    yaw = np.arctan2(dy, dx) * 180 / np.pi - 90.0 + roll # cancel out the roll

    return yaw

def head_pose(landmarks: list) -> tuple:
    """
    Classifies the intensity and direction of the pitch, roll, and yaw of the head

    :param landmarks: landmarks of the face
    :return: tuple of the classification of the pose
    """
    pitch = calc_pitch(landmarks)
    roll = calc_roll(landmarks)
    yaw = calc_yaw(landmarks, roll)

    pitch_intensity = classify_pose(pitch)
    roll_intensity = classify_pose(roll)
    yaw_intensity = classify_pose(yaw)

    pitch_direction = ("up" if pitch > 0 else "down") if pitch_intensity != "center" else "neutral"
    roll_direction = ("left" if roll > 0 else "right") if roll_intensity != "center" else "neutral"
    yaw_direction = ("left" if yaw > 0 else "right") if yaw_intensity != "center" else "neutral"

    return ("pitch", pitch_direction, pitch_intensity), ("roll", roll_direction, roll_intensity), ("yaw", yaw_direction, yaw_intensity)

def eye_aspect_ratio(landmarks: list, eye: str) -> float:
    """
    Computes the eye aspect ratio for the given eye

    :param landmarks: landmarks of the face
    :param eye: eye to compute the eye aspect ratio for
    :return: eye aspect ratio for the given eye
    """
    eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]) if eye == "left" else np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_ar = (A + B) / (2.0 * C)
    return eye_ar

def get_face_bounding_box(landmarks):
    """
    Computes the bounding box of the face

    :param landmarks: landmarks of the face
    :return: bounding box of the face
    """
    x_min = min(landmark.x for landmark in landmarks.parts())
    y_min = min(landmark.y for landmark in landmarks.parts())
    x_max = max(landmark.x for landmark in landmarks.parts())
    y_max = max(landmark.y for landmark in landmarks.parts())
    
    return x_min, y_min, x_max, y_max

def facial_centroid(landmarks):
    """
    Computes the centroid of the face

    :param landmarks: landmarks of the face
    :return: centroid of the face
    """
    x = [landmark.x for landmark in landmarks.parts()]
    y = [landmark.y for landmark in landmarks.parts()]
    return (sum(x) / len(x), sum(y) / len(y))

def facial_occupancy_area(landmarks, image_dims):
    """
    Computes the area that the image occupies in the frame

    :param landmarks: landmarks of the face
    :param image_dims: dimensions of the image
    :return: ratio of the face area to the image area
    """
    x_min, y_min, x_max, y_max = get_face_bounding_box(landmarks)
    face_width = x_max - x_min
    face_height = y_max - y_min
    return (face_width * face_height) / (image_dims[0] * image_dims[1])

def crop_frame(frame, landmarks, output_size=(224, 224), padding_factor=2):
    """
    Crops the frame to the face and resizes it to the output size
    Adds padding to the face to include more context
    Uses the facial centroid to center the face in the cropped frame
    Uses only the face height for cropping and ensures a square aspect ratio

    :param frame: frame to crop
    :param landmarks: landmarks of the face
    :param output_size: size of the output frame
    :param padding_factor: factor to pad the face
    :return: cropped and resized frame
    """
    _, y_min, _, y_max = get_face_bounding_box(landmarks)
    
    face_height = y_max - y_min
    face_width = face_height  # Use face height for width to ensure square aspect ratio
    
    padding = int((face_height * (padding_factor - 1)) / 2)
    
    centroid_x, centroid_y = facial_centroid(landmarks)
    
    crop_x_min = max(0, int(centroid_x - (face_width / 2) - padding))
    crop_y_min = max(0, int(centroid_y - (face_height / 2) - padding))
    crop_x_max = min(frame.shape[1], int(centroid_x + (face_width / 2) + padding))
    crop_y_max = min(frame.shape[0], int(centroid_y + (face_height / 2) + padding))
    
    cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    resized_frame = cv2.resize(cropped_frame, output_size)
    
    return resized_frame
    
def frame_brightness(frame):
    """
    Computes the brightness of the frame

    :param frame: frame to compute the brightness for
    :return: brightness of the frame
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_frame)

def is_blurry(image, threshold=100):
    """
    Determines if an image is blurry based on the variance of the Laplacian
    
    :param image: image to check for blurriness
    :param threshold: threshold for the variance of the Laplacian
    :return: whether the image is blurry
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold

def display_timeout(frame, start, total):
    """
    Displays the time since the last capture on the frame

    :param frame: frame to display the time on
    :param start: start time of the timeout
    :param total: total time of the timeout
    """
    elapsed = time.time() - start
    bar_width = int(frame.shape[1] * 0.8)
    bar_height = 10
    fill_width = min(int((elapsed / total) * bar_width), bar_width)
    
    pt1 = (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1))
    pt2 = (int(frame.shape[1] * 0.1) + fill_width, int(frame.shape[0] * 0.1) + bar_height)
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), -1)
    
    if fill_width < bar_width:
        pt1 = (int(frame.shape[1] * 0.1) + fill_width, int(frame.shape[0] * 0.1))
        pt2 = (int(frame.shape[1] * 0.1) + bar_width, int(frame.shape[0] * 0.1) + bar_height)
        cv2.rectangle(frame, pt1, pt2, (0, 0, 0), -1)

if __name__ == "__main__":
    CURRENT_LOCATION = 'living room'
    GLASSES = False
    CONTACTS = False
    
    LABELS_PATH = 'dataset/labels.txt'
    ENV_PROG_PATH = 'dataset/env_prog.txt'

    TOTAL_POINTS = 17 * 10 # 17 points across, 10 points down
    PROGRESS = 0

    CAPTURE_TIMEOUT = 0.5 # seconds

    try:
        # initialize video capture and the window
        video = cv2.VideoCapture(0)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # initialize facial landmark detector and predictor
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        capturing = False # indicates the program is in a capturing session
        next_point = False # set at the end of a capturing session to move to the next point, then reset
        key_pressed = 0 # captures the key pressed by the user, 32 is space, q is quit

        # variables for process to set a timeout for capturing
        timeout_event = multiprocessing.Event()
        timeout_process = multiprocessing.Process(target=set_timeout, args=(timeout_event, CAPTURE_TIMEOUT))
        timeout_process.start()
        timeout_time_start = None # time the timeout started

        # get the current point and last label index from the environment progress file
        current_point, last_label_i = get_current_progress(ENV_PROG_PATH)
        points_matrix = None # matrix to hold the coordinates of each point on screen

        images = [] # batch of images to save
        labels = [] # batch of labels to save

        while not (exitting := False):
            # capture the frame
            ret, frame = video.read()
            if not ret:
                print("Error capturing frame. Exiting...")
                break

            # initialize the points matrix on the first iteration
            if not points_matrix:
                points_matrix = [[0] * 17 for _ in range(10)]
                h, w = frame.shape[:2] # initialized inside of the loop so we can use the frame's dimensions
                for x in range(0, 17):
                    for y in range(0, 10):
                        points_matrix[y][x] = (x * w / 16, y * h / 9)

            # draw the current point on the frame
            x, y = int(points_matrix[current_point[1]][current_point[0]][0]), int(points_matrix[current_point[1]][current_point[0]][1])
            cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)

            # display the time since the last capture
            if capturing and timeout_event.is_set():
                if not timeout_time_start: # only set the start time once
                    timeout_time_start = time.time()
                display_timeout(frame, timeout_time_start, CAPTURE_TIMEOUT)
            else:
                timeout_time_start = None # reset so the start time can be set again on the next capture

            # convert the frame to grayscale and detect faces
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (is_blurry(frame, 10)): # threshold of 10 seems low, but it's empirically determined to work well
                continue # skip blurry frames
            
            # detect faces in the frame and get the landmarks
            faces = dlib_detector(gray_frame, 0)
            for face in faces: # only one face is expected
                # get the landmarks of the face and draw them on the frame
                landmarks = dlib_predictor(gray_frame, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

                # if capturing, get the next image and label
                if capturing and not timeout_event.is_set():
                    # compute features for label's metadata
                    left_eye_ar = eye_aspect_ratio(landmarks, "left")
                    right_eye_ar = eye_aspect_ratio(landmarks, "right")
                    head_position = head_pose(landmarks)
                    brightness = frame_brightness(frame)
                    facial_occ_area = facial_occupancy_area(landmarks, frame.shape[:2])

                    # make the label and crop the frame
                    label = make_label(
                        current_point,
                        f"dataset/imgs/{last_label_i}.jpg",
                        (224, 224),
                        CURRENT_LOCATION,
                        brightness,
                        (left_eye_ar, right_eye_ar),
                        head_position,
                        facial_occ_area,
                        GLASSES,
                        CONTACTS,
                        str(datetime.datetime.now().isoformat())
                    )
                    cropped = crop_frame(gray_frame, landmarks)
                    last_label_i += 1

                    images.append(cropped)
                    labels.append(label)

                    timeout_event.set() # wait for the timeout to expire before capturing again
    
                # all points captured for the current location, cleanup, reset current point, and exit
                if (PROGRESS == TOTAL_POINTS):
                    print(f"All points captured for {CURRENT_LOCATION}. Exiting...")
                    save_current_progress(ENV_PROG_PATH, (0, 0), last_label_i) # keep label index across runs
                    exitting = True
                    break

            # once a capturing session is done, set the next point to be viewed
            if next_point:
                PROGRESS += 1
                next_point = False
                current_point = (current_point[0] + 1, current_point[1])
                if current_point[0] >= 17:
                    current_point = (0, current_point[1] + 1)
                if current_point[1] >= 10:
                    current_point = (0, 0)

            # show the frame and check for key presses
            cv2.imshow("Frame", frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 32 and not capturing: # if not capturing, 'space' to start capturing
                capturing = True
            elif key_pressed == 32 and capturing: # if capturing, 'space' to stop capturing
                key_pressed = cv2.waitKey(0) & 0xFF # wait until the user saves or discards the images
                if key_pressed == ord("s"):
                    # save images and labels
                    for i, (img, label) in enumerate(zip(images, labels)):
                        cv2.imwrite(f"dataset/imgs/{last_label_i - len(images) + i}.jpg", img)
                        with open(LABELS_PATH, 'a') as file:
                            file.write(f"{label}\n")
                    next_point = True # set the next point to be viewed
                    capturing = False # exit capturing session
                elif key_pressed == 32: # 'space' to discard the batch and try again
                    images = [] # discard images
                    labels = [] # discard labels
            elif key_pressed == ord("q"): # 'q' to quit the program
                break

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = traceback.extract_tb(exc_traceback)
        
        filename = traceback_details[-1][0]
        line_number = traceback_details[-1][1]
        print(f"An error occurred: {e} at line {line_number} in {filename}")

    finally:
        # save progress and cleanup resources
        save_current_progress(ENV_PROG_PATH, current_point, last_label_i)
        video.release()
        cv2.destroyAllWindows()


"""
DON'T FORGET TO UPDATE FOR LOCATION, GLASSES, AND CONTACTS
"""