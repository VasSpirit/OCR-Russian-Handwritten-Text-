import glob
import easyocr
reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
for p in sorted(glob.glob('/tmp/probes/*.png')):
    try:
        res = reader.readtext(p, paragraph=False)
        txt = ' | '.join(t[1] for t in res)
        conf = round(sum(t[2] for t in res)/max(1, len(res)), 3)
        print(p.split('/')[-1], 'n=', len(res), 'conf=', conf, 'txt=', repr(txt[:80]))
    except Exception as e:
        print(p.split('/')[-1], 'ERR', type(e).__name__, e)
