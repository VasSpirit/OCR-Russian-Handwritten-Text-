import os

from PIL import Image, ImageDraw

probe_dir = '/tmp/probes'
paths = sorted(
    os.path.join(probe_dir, x)
    for x in os.listdir(probe_dir)
    if x.endswith('.png')
)
items = [(os.path.basename(p), Image.open(p).convert('RGB')) for p in paths]

Wmax = max(im.width for _, im in items) + 10
H = 220
ncol = 2
nrow = -(-len(items) // ncol)
canvas = Image.new('RGB', (Wmax * ncol, H * nrow), 'white')
draw = ImageDraw.Draw(canvas)


for i, (name, im) in enumerate(items):
    col = i % ncol
    row = i // ncol
    x = col * Wmax +  5
    y = row * H +  5
    canvas.paste(im, (x, y))
    draw.text((x, y -  15), name, fill='red')

canvas = canvas.resize((canvas.width //  2, canvas.height //  2))
canvas.save('/tmp/probes_montage.png')
print('saved', canvas.size)