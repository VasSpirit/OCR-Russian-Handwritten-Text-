import re
import cv2
import numpy as np
from PIL import Image

BASE = '/workspace/project/OCR-Russian-Handwritten-Text-/tmp_data/'
img = cv2.imread('/tmp/page_embedded.png', cv2.IMREAD_GRAYSCALE)

cols_raw = open(BASE + 'cols.txt', encoding='utf-8').read()
rows_raw = open(BASE + 'rows.txt', encoding='utf-8').read()
cols = [int(x) for x in re.findall(r"\d+", cols_raw)]
rows_c = [int(x) for x in re.findall(r"\d+", rows_raw)]

print('cols', cols)
print('rows_c', rows_c)

bounds = []
for i in range(8):
    if i ==  0:
        left = cols[0]
    else:
        left = cols[2 * i +  1]
    if i ==  7:
        right = cols[15]
    else:
        right = cols[2 * i +  2]
    bounds.append((int(left), int(right)))

print('bounds')
for i, b in enumerate(bounds):
    print(i, b)

for ci, (x0, x1) in enumerate(bounds):
    for ri, yc in enumerate(rows_c):
        y0 = yc -  55
        y1 = yc +  55
        crop = img[max(0,y0):y1, max(0,x0):x1]
        if ci ==  1 and ri ==  0:
            cv2.imwrite('/tmp/cell_probe_1.png', crop)
        if ci ==  2and ri ==  0:
            cv2.imwrite('/tmp/cell_probe_2.png', crop)
        if ci ==  1and ri ==  3:
            cv2.imwrite('/tmp/cell_probe_3.png', crop)

print('done')
