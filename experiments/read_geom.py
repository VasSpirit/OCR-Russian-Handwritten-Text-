import re
raw = open('/workspace/project/OCR-Russian-Handwritten-Text-/tmp_data/cols.txt', encoding='utf-8').read()
nums = re.findall(r'\d+', raw)
print(nums)