import re
import cv2
import os

BASE = '/workspace/project/OCR-Russian-Handwritten-Text-/tmp_data/'
img = cv2.imread('/tmp/page_embedded.png', cv2.IMREAD_GRAYSCALE)
rows_c = [int(x) for x in re.findall(r'\d+', open(BASE + 'rows.txt', encoding='utf-8').read())]

bounds = [
    (36, 128),
    (744, 868),
    (872, 1008),
    (1012, 1112),
    (1116, 1308),
    (1312, 1544),
    (2068, 2368),
    (2372, 2372),
]

os.makedirs('/tmp/probes', exist_ok=True)

probes = [
    ('street_r1', 1,  0),
    ('house_r1',  2,  0),
    ('num_r1',  3,  0),
    ('gen_age_r1',  4,  0),
    ('contact_r1',  5,  0),
    ('comment_r1',  6,  0),
    ('street_r2',  1,  1),
    ('contact_r3',  5,  2),
    ('apart_r3',  3,  2),
    ('street_last',  1,  25),
]

for tag, ci, ri in probes:

    x0,  x1 = bounds[ci]
    y0 = rows_c[ri] - 55
    y1 = rows_c[ri] + 55
    crop = img[max(0, y0):y1, x0:x1]
    p = f'/tmp/probes/{tag}.png'
    cv2.imwrite(p, crop)
    print(tag, p, crop.shape)