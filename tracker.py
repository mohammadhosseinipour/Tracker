import cv2
import numpy as np

# Initialize SIFT detector with parameters to detect more keypoints
sift=cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)


# Parameters for Lucas-Kanade optical flow
lk_params=dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# List to store the keypoints and descriptors
keypoints_list=[]
descriptors_list=[]
old_points=[]
rois=[]

def select_roi(event, x, y, flags, param):
    global selecting, roi_start, roi_end, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting=True
        roi_start=(x, y)
        roi_end=(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            roi_end=(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting=False
        roi_end=(x, y)
        x1, y1=roi_start
        x2, y2=roi_end
        roi=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        add_roi_keypoints(roi, frame)

def add_roi_keypoints(roi, frame):
    global keypoints_list, descriptors_list, old_points, rois
    x, y, w, h=roi
    roi_frame=frame[y:y+h, x:x+w]
    keypoints, descriptors=sift.detectAndCompute(roi_frame, None)
    keypoints=[cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
    if keypoints:
        old_points.append(np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2))
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        rois.append(roi)
    else:
        old_points.append(None)

def filter_keypoints(roi, points):
    x, y, w, h=roi
    new_points=[]
    for pt in points:
        px, py=pt.ravel()
        if x <= px <= x + w and y <= py <= y + h:
            new_points.append(pt)
    return np.array(new_points, dtype=np.float32).reshape(-1, 1, 2)

selecting=False
roi_start=(0, 0)
roi_end=(0, 0)

cap=cv2.VideoCapture(0)
cv2.namedWindow("Webcam Feed")
cv2.setMouseCallback("Webcam Feed", select_roi)

ret, frame=cap.read()
old_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame=cap.read()
    if not ret:
        break

    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selecting:
        x1, y1=roi_start
        x2, y2=roi_end
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    new_points_list=[]
    for old_pts in old_points:
        if old_pts is not None:
            new_pts, status, err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_pts, None, **lk_params)
            if new_pts is not None and status is not None:
                status=status.reshape(-1)
                good_new=new_pts[status == 1]
                new_points_list.append(good_new)
            else:
                new_points_list.append(None)
        else:
            new_points_list.append(None)

    updated_rois=[]
    updated_keypoints_list=[]
    updated_descriptors_list=[]
    updated_old_points=[]

    for i, (roi, new_pts) in enumerate(zip(rois, new_points_list)):
        if new_pts is not None and new_pts.size > 0:
            new_pts=filter_keypoints(roi, new_pts)
            if new_pts.size > 0:
                x, y, w, h=roi
                mean_position=np.mean(new_pts, axis=0).reshape(-1)
                cx, cy=mean_position

                # Calculate the new top-left corner of the ROI
                new_x=int(cx - w / 2)
                new_y=int(cy - h / 2)
                new_roi=(new_x, new_y, w, h)
                updated_rois.append(new_roi)

                cv2.rectangle(frame, (new_roi[0], new_roi[1]), (new_roi[0] + new_roi[2], new_roi[1] + new_roi[3]), (255, 0, 0), 2)
                for pt in new_pts:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)

                add_roi_keypoints(new_roi, frame)

                updated_keypoints_list.append(keypoints_list[i])
                updated_descriptors_list.append(descriptors_list[i])
                updated_old_points.append(new_pts)
            else:
                old_points[i]=None

    rois=updated_rois
    keypoints_list=updated_keypoints_list
    descriptors_list=updated_descriptors_list
    old_points=updated_old_points

    cv2.imshow("Webcam Feed", frame)

    old_gray=frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
