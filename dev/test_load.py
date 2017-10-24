from PIL import Image
import numpy as np
import cv2
img = Image.open('../MyData9/IMG/center_2017_10_03_22_49_13_493.jpg')
print(img)
# RGB

r, g, b = img.split()
img = Image.merge("RGB", (b, g, r))

print(img)
img = np.asarray(img)
print(img)
# print(img[:, :, 1])
# BGR
img2 = cv2.imread('../MyData9/IMG/center_2017_10_03_22_49_13_493.jpg')
# print(img2[:, :, 1])
#
# neq = np.not_equal(img[:, :, 1], img2[:, :, 1])
# rows, cols = neq.shape
# for row in range(rows):
#     for col in range(cols):
#         if bool(neq[row, col]) is not False:
#             print("r: ", row, ", col: ", col, ", ", neq[row, col])
#
# #print(img[129, 294, 1])
#
# print(img[129, 293, 1])
# print(img2[129, 293, 1])
