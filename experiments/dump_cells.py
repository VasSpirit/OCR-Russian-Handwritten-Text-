import cv2
import re
import os
rows_c=[int(x) for x in re.findall(r'\d+',open('experiments/data/rows.txt',encoding='utf-8').read())]
cols=[('street',36,128),('house',744,868),('apartment',1012,1112),('gender_age',1116,1308),('contact_result',1312,1544),('comment',1544,2068)]
img=cv2.imread('/tmp/page_full.png',cv2.IMREAD_GRAYSCALE)
os.makedirs('/tmp/cells',exist_ok=True)
n=0
for ri,rc in enumerate(rows_c[:4]):
    y0=rc-60
    y1=rc+60
    for name,x0,x1 in cols:
        crop=img[y0:y1,x0:x1]
        ink=(crop<128).sum()
        mean=crop.mean()
        p=os.path.join('/tmp/cells',f'r{ri+1}_{name}.png')
        cv2.imwrite(p,crop)
        n+=1
        print(p,round(mean,1),int(ink))
print('total',n)