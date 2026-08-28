#!/usr/bin/env python3
import sys
from pathlib import Path

def fix(src_text):
    lines = src_text.split('\n')
    out = []
    for line in lines:
        a = line.replace('\u02d8', '')
        a = a.replace('\uff09', ')')
        a = a.replace('\uff08', '(')
        a = a.strip()
        nopen = a.count('(') 
        nclose = a.count(')')
        if nopen > nclose:
            a += ')' * (nopen - nclose)
        out.append(a)
    result = '\n'.join(out)
    return result

def main(args):
    for path in args:
        p = Path(path)
        t = p.read_text(encoding='utf-8')
        p.write_text(fix(t), encoding='utf-8')
        print('fixed', path)

if __name__ == '__main__':
    main(sys.argv[1:])