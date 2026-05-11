import requests, json, io
from PIL import Image

# 1. Upload Excel first
with open('f:/face-pj/员工信息导入范本.xlsx', 'rb') as f:
    r = requests.post('http://localhost:8000/api/upload-excel/', files={'file': f})
    print('Excel import:', r.json())

# 2. Check employees
r = requests.get('http://localhost:8000/api/employees/')
emps = r.json()
print(f'Employees: {len(emps)}')
for e in emps:
    print(f'  {e["name"]} ({e["employee_id"]})')

# 3. Test upload with photo named after employee
for emp_name in ['张三', '李四']:
    img = Image.new('RGB', (200, 200), color='blue')
    buf = io.BytesIO()
    img.save(buf, 'JPEG')
    buf.seek(0)
    r = requests.post('http://localhost:8000/api/batch-link-photos/',
                      files=[('photos', (f'{emp_name}.jpg', buf, 'image/jpeg'))])
    print(f'{emp_name}.jpg:', json.dumps(r.json(), ensure_ascii=False))
