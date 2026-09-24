import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_landmark_coords(landmarks, landmark_enum, mp_pose):
    lm = landmarks[mp_pose.PoseLandmark[landmark_enum].value]
    return [lm.x, lm.y], lm.visibility


def get_knee_angle(landmarks, mp_pose, side="left", visibility_threshold=0.6):
    side = side.upper()
    hip, hip_vis = get_landmark_coords(landmarks, f"{side}_HIP", mp_pose)
    knee, knee_vis = get_landmark_coords(landmarks, f"{side}_KNEE", mp_pose)
    ankle, ankle_vis = get_landmark_coords(landmarks, f"{side}_ANKLE", mp_pose)

    is_reliable = (
        hip_vis > visibility_threshold and
        knee_vis > visibility_threshold and
        ankle_vis > visibility_threshold
    )

    angle = calculate_angle(hip, knee, ankle)
    return angle, is_reliable