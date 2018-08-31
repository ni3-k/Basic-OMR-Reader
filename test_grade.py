import cv2
import numpy as np
import argparse
from imutils import perspective
from imutils import contours


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

ANSWER = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
options = 5		# Options available for answer

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

_, cnts, h = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) > 0:
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*peri, True)
		if len(approx) == 4:
			docCnt = approx
			break


original = perspective.four_point_transform(image, docCnt.reshape(4, 2))
gray = perspective.four_point_transform(gray, docCnt.reshape(4,2))
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]

_, cnts, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
mcq = []
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w/float(h)

	if w>=20 and h>=20 and ar>=0.9 and ar<=1.1:
		mcq.append(c)


mcq = contours.sort_contours(mcq, method="top-to-bottom")[0]  # returns contour, bounding box
correct = 0
for (q, i) in enumerate(np.arange(0, len(mcq), options)):
	cnts = contours.sort_contours(mcq[i:i+options], method="left-to-right")[0]

	bubbled = None
	for (j, c) in enumerate(cnts):
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)
		print(total)
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)
			print(bubbled)
	
	color = (0, 0, 255)
	answer = ANSWER[q]
	print(bubbled[1], "answer = ", answer)

	if bubbled[1] == answer:
		color = (0, 255, 0)
		correct += 1
	cv2.drawContours(original, [cnts[answer]], -1, color, 2)


score = correct/float(options) * 100
cv2.putText(original, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


cv2.imshow("input", image)
cv2.imshow("output", original)

cv2.waitKey(0)
cv2.destroyAllWindows()