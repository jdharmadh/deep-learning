import cv2
import numpy as np

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    return (dx1*dx2 + dy1*dy2) / np.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2) + 1e-10)

def draw_squares(image, squares):
    for sq in squares:
        shift = 1
        r = cv2.boundingRect(np.array(sq))
        r = (r[0] + r[2] // 4, r[1] + r[3] // 4, r[2] // 2, r[3] // 2)

        roi = image[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        color = tuple(map(int, cv2.mean(roi)[:3]))
        cv2.polylines(image, [np.array(sq)], True, color, 2, cv2.LINE_AA, shift)

        center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
        axes = (r[2] // 2, r[3] // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)

def find_squares(image, inv=False):
    squares = []
    gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.GaussianBlur(gray0, (7, 7), 1.5)
    gray = cv2.Canny(gray0, 0, 30, apertureSize=3)

    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 9, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 5 and cv2.isContourConvex(approx):
            max_cosine = 0
            approx = approx.reshape(-1, 2)
            for j in range(2, 5):
                cosine = abs(angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                max_cosine = max(max_cosine, cosine)
            if max_cosine < 0.3:
                squares.append(approx.tolist())
    return squares

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squares = find_squares(frame)
        draw_squares(frame, squares)
        cv2.imshow("Rubic Detection Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
