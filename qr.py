import cv2
import numpy as np
from collections import Counter

GRID_SIZE = 6
WINDOW_SIZE = 500
SQUARE_SIZE = WINDOW_SIZE // GRID_SIZE

COLOR_RANGES = {
    "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    "blue": [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
    "green": [(np.array([35, 100, 100]), np.array([85, 255, 255]))]
}

def get_largest_square_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 50000:  # reject small contours
                return np.array([p[0] for p in approx], dtype="float32")
    return None

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def perspective_transform(corners):
    ordered = order_points(corners)
    dst = np.array([
        [0, 0],
        [WINDOW_SIZE - 1, 0],
        [WINDOW_SIZE - 1, WINDOW_SIZE - 1],
        [0, WINDOW_SIZE - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return M

def detect_dominant_color(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hsv_reshaped = hsv.reshape(-1, 3)
    counts = Counter()
    for pixel in hsv_reshaped:
        for color, ranges in COLOR_RANGES.items():
            for lower, upper in ranges:
                if all(lower <= pixel) and all(pixel <= upper):
                    counts[color] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return "unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    grid_corners = get_largest_square_contour(frame)
    color_map = {"red": 1, "blue": 2, "black": 3, "green": 4}
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    if grid_corners is not None:
        M = perspective_transform(grid_corners)
        warped = cv2.warpPerspective(frame, M, (WINDOW_SIZE, WINDOW_SIZE))

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x0 = x * SQUARE_SIZE
                y0 = y * SQUARE_SIZE
                roi = warped[y0:y0+SQUARE_SIZE, x0:x0+SQUARE_SIZE]
                color = detect_dominant_color(roi)
                cv2.putText(warped, color, (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.rectangle(warped, (x0, y0), (x0+SQUARE_SIZE, y0+SQUARE_SIZE), (255, 255, 255), 1)
                grid[y, x] = color_map.get(color, 0)

        print(grid)

        cv2.imshow("Warped Grid", warped)
    else:
        cv2.putText(display, "Grid not found", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera Feed", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()