import requests, io, json
from PIL import Image
import openpyxl

wb = openpyxl.Workbook()
ws = wb.active
ws.append(['name', 'department', 'title', 'hire_date', 'employment_status', 'industry'])
ws.append(['安岷', '事业部', '老师', '2015-03-01', '在职', '教育'])
ws.append(['杜晶晶', '事业部', '老师', '2016-07-01', '在职', '教育'])
ws.append(['罗树忠', '技术部', '老师', '2014-01-15', '在职', '教育'])
ws.append(['汪学明', '销售部', '老师', '2017-09-01', '在职', '教育'])
ws.append(['贺君宏', '技术部', '老师', '2018-05-01', '在职', '教育'])
excel_buf = io.BytesIO()
wb.save(excel_buf)
excel_buf.seek(0)
r = requests.post('http://localhost:8000/api/upload-excel/', files={'file': ('test.xlsx', excel_buf)})
print('Excel:', r.json()['message'])

test_files = [
    '安岷老师.jpg',
    '03-安岷老师(1).jpg',
    '杜晶晶-事业部.jpg',
    '罗树忠.png',
    '贺君宏_照片.jpeg',
]

for fname in test_files:
    img = Image.new('RGB', (200, 200), color='blue')
    buf = io.BytesIO()
    img.save(buf, 'JPEG')
    buf.seek(0)
    r = requests.post('http://localhost:8000/api/batch-link-photos/',
                      files=[('photos', (fname, buf, 'image/jpeg'))])
    result = r.json()
    if result['matched']:
        m = result['matched'][0]
        print(f'OK  {fname} -> {m["name"]} (face={m["has_face"]})')
    elif result['unmatched']:
        print(f'NO  {fname} -> {result["unmatched"][0]["reason"]}')
    elif result['errors']:
        print(f'ERR {fname} -> {result["errors"][0]["error"][:60]}')
    else:
        print(f'??? {fname}')
