from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import shutil
from datetime import date
from . import crud, schemas
from .database import get_db
from .face_service import FaceRecognitionService

router = APIRouter()
face_service = FaceRecognitionService()
PHOTOS_DIR = os.getenv("PHOTOS_DIR", "./photos")

@router.post("/employees/", response_model=schemas.EmployeeResponse)
def create_employee(
    employee_id: str,
    name: str,
    department: str = None,
    title: str = None,
    hire_date: str = None,
    employment_status: str = None,
    industry: str = None,
    photo: UploadFile = None,
    db: Session = Depends(get_db)
):
    db_employee = crud.get_employee(db, employee_id=employee_id)
    if db_employee:
        raise HTTPException(status_code=400, detail="员工ID已存在")
    
    photo_path = None
    face_embedding = None
    
    if photo:
        if not photo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="只支持PNG和JPG格式的照片")
        
        photo_filename = f"{employee_id}.jpg"
        photo_path = os.path.join(PHOTOS_DIR, photo_filename)
        
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
        
        face_embedding = face_service.extract_embedding(photo_path)
    
    hire_date_obj = date.fromisoformat(hire_date) if hire_date else None
    
    employee_create = schemas.EmployeeCreate(
        employee_id=employee_id,
        name=name,
        department=department,
        title=title,
        hire_date=hire_date_obj,
        employment_status=employment_status,
        industry=industry
    )
    
    return crud.create_employee(db=db, employee=employee_create, 
                                photo_path=photo_path, face_embedding=face_embedding)

@router.get("/employees/", response_model=list[schemas.EmployeeResponse])
def read_employees(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    employees = crud.get_employees(db, skip=skip, limit=limit)
    return employees

@router.get("/employees/{employee_id}", response_model=schemas.EmployeeResponse)
def read_employee(employee_id: str, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id=employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="员工不存在")
    return db_employee

@router.put("/employees/{employee_id}", response_model=schemas.EmployeeResponse)
def update_employee(
    employee_id: str,
    name: str = None,
    department: str = None,
    title: str = None,
    hire_date: str = None,
    employment_status: str = None,
    industry: str = None,
    db: Session = Depends(get_db)
):
    db_employee = crud.get_employee(db, employee_id=employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="员工不存在")
    
    hire_date_obj = date.fromisoformat(hire_date) if hire_date else None
    
    employee_update = schemas.EmployeeUpdate(
        name=name,
        department=department,
        title=title,
        hire_date=hire_date_obj,
        employment_status=employment_status,
        industry=industry
    )
    
    return crud.update_employee(db=db, db_employee=db_employee, employee_update=employee_update)

@router.delete("/employees/{employee_id}")
def delete_employee(employee_id: str, db: Session = Depends(get_db)):
    success = crud.delete_employee(db, employee_id=employee_id)
    if not success:
        raise HTTPException(status_code=404, detail="员工不存在")
    return {"message": "删除成功"}

@router.post("/search/")
def search_employees(
    name: str = None,
    department: str = None,
    employment_status: str = None,
    industry: str = None,
    hire_date_from: str = None,
    hire_date_to: str = None,
    min_service_years: int = None,
    db: Session = Depends(get_db)
):
    hire_date_from_obj = date.fromisoformat(hire_date_from) if hire_date_from else None
    hire_date_to_obj = date.fromisoformat(hire_date_to) if hire_date_to else None

    if min_service_years:
        from datetime import date as date_cls
        cutoff = date_cls.today().replace(year=date_cls.today().year - min_service_years)
        hire_date_to_obj = cutoff

    employees = crud.search_employees(
        db,
        name=name,
        department=department,
        employment_status=employment_status,
        industry=industry,
        hire_date_from=hire_date_from_obj,
        hire_date_to=hire_date_to_obj
    )
    return {"results": [schemas.EmployeeResponse.model_validate(e) for e in employees]}

def _verify_matches(input_embedding, employees, threshold, gap_ratio,
                   employment_status, department, industry, cutoff_date):
    scored = []
    for employee in employees:
        if not employee.face_embedding:
            continue
        if employment_status and employee.employment_status != employment_status:
            continue
        if department and employee.department != department:
            continue
        if industry and employee.industry != industry:
            continue
        if cutoff_date and employee.hire_date:
            if employee.hire_date > cutoff_date:
                continue

        sim = face_service.compare_faces(input_embedding, employee.face_embedding)
        scored.append((employee, round(float(sim), 4)))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < threshold:
        return []

    top = scored[0][1]
    second = scored[1][1] if len(scored) > 1 else 0.0

    if second > 0 and top / second < gap_ratio:
        return []

    results = []
    for emp, sim in scored:
        if sim < threshold:
            break
        results.append({
            "employee_id": emp.employee_id,
            "name": emp.name,
            "department": emp.department,
            "title": emp.title,
            "similarity": sim,
            "photo_path": emp.photo_path,
            "employment_status": emp.employment_status,
            "hire_date": emp.hire_date
        })
    return results

@router.post("/face-match/", response_model=list[schemas.FaceMatchResult])
def face_match(
    photo: UploadFile = File(...),
    threshold: float = 0.35,
    gap_ratio: float = 1.5,
    employment_status: str = None,
    department: str = None,
    industry: str = None,
    min_service_years: int = None,
    db: Session = Depends(get_db)
):
    if not photo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="只支持PNG和JPG格式的照片")
    
    temp_path = os.path.join(PHOTOS_DIR, "temp_match.jpg")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
        
        input_embedding = face_service.extract_embedding(temp_path)
        if not input_embedding:
            return []

        employees = crud.get_all_employees_with_embedding(db)

        from datetime import date as date_cls
        cutoff_date = None
        if min_service_years:
            cutoff_date = date_cls.today().replace(year=date_cls.today().year - min_service_years)

        results = _verify_matches(input_embedding, employees, threshold, gap_ratio,
                                  employment_status, department, industry, cutoff_date)
        return results[:5]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/batch-face-match/")
def batch_face_match(
    photos: list[UploadFile] = File(...),
    threshold: float = 0.35,
    gap_ratio: float = 1.5,
    employment_status: str = None,
    department: str = None,
    industry: str = None,
    min_service_years: int = None,
    db: Session = Depends(get_db)
):
    if not photos:
        raise HTTPException(status_code=400, detail="请上传照片文件")

    employees = crud.get_all_employees_with_embedding(db)

    from datetime import date as date_cls
    cutoff_date = None
    if min_service_years:
        cutoff_date = date_cls.today().replace(year=date_cls.today().year - min_service_years)

    results = []

    for idx, photo in enumerate(photos):
        if not photo.filename or not photo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        temp_path = os.path.join(PHOTOS_DIR, f"temp_batch_{idx}.jpg")
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

            face_count = face_service.count_faces(temp_path)
            input_embedding = face_service.extract_embedding(temp_path)
            if not input_embedding:
                results.append({
                    "filename": photo.filename,
                    "face_count": face_count,
                    "matches": [],
                    "error": "无法提取人脸特征"
                })
                continue

            matches = _verify_matches(input_embedding, employees, threshold, gap_ratio,
                                      employment_status, department, industry, cutoff_date)
            results.append({
                "filename": photo.filename,
                "face_count": face_count,
                "matches": matches[:5]
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return {
        "total_photos": len(photos),
        "results": results
    }

@router.post("/multi-face-match/")
def multi_face_match(
    photo: UploadFile = File(...),
    threshold: float = 0.40,
    gap_ratio: float = 1.5,
    employment_status: str = None,
    department: str = None,
    industry: str = None,
    min_service_years: int = None,
    db: Session = Depends(get_db)
):
    if not photo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="只支持PNG和JPG格式的照片")

    temp_path = os.path.join(PHOTOS_DIR, "temp_multi.jpg")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        face_data = face_service.extract_embeddings(temp_path)
        if not face_data:
            raise HTTPException(status_code=400, detail="未检测到人脸")

        employees = crud.get_all_employees_with_embedding(db)

        from datetime import date as date_cls
        cutoff_date = None
        if min_service_years:
            cutoff_date = date_cls.today().replace(year=date_cls.today().year - min_service_years)

        face_results = []
        for fi, fd in enumerate(face_data):
            matches = _verify_matches(fd["embedding"], employees, threshold, gap_ratio,
                                      employment_status, department, industry, cutoff_date)
            face_results.append({
                "face_index": fi + 1,
                "bbox": fd["bbox"],
                "top_match": matches[0] if matches else None,
                "matches": matches[:5],
                "is_unknown": len(matches) == 0
            })

        known = [f for f in face_results if not f["is_unknown"]]

        return {
            "total_faces": len(face_data),
            "known_faces": len(known),
            "unknown_faces": len(face_data) - len(known),
            "faces": face_results
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="只支持xlsx格式的Excel文件")
    
    try:
        import pandas as pd
        
        df = pd.read_excel(file.file)
        
        required_columns = ['name']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Excel文件必须包含 name（姓名）列")
        
        created_count = 0
        updated_count = 0
        
        for _, row in df.iterrows():
            name = str(row.get('name', '')).strip()

            employee_id = str(row.get('employee_id', '')).strip() if pd.notna(row.get('employee_id')) else ''
            if not employee_id:
                employee_id = f"AUTO_{hash(name) % 900000 + 100000}"

            if not name:
                continue
            
            db_employee = crud.get_employee(db, employee_id=employee_id)
            
            if db_employee:
                employee_update = schemas.EmployeeUpdate(
                    name=name,
                    department=str(row.get('department', '')).strip() if pd.notna(row.get('department')) else None,
                    title=str(row.get('title', '')).strip() if pd.notna(row.get('title')) else None,
                    hire_date=row.get('hire_date'),
                    employment_status=str(row.get('employment_status', '')).strip() if pd.notna(row.get('employment_status')) else None,
                    industry=str(row.get('industry', '')).strip() if pd.notna(row.get('industry')) else None
                )
                crud.update_employee(db=db, db_employee=db_employee, employee_update=employee_update)
                updated_count += 1
            else:
                employee_create = schemas.EmployeeCreate(
                    employee_id=employee_id,
                    name=name,
                    department=str(row.get('department', '')).strip() if pd.notna(row.get('department')) else None,
                    title=str(row.get('title', '')).strip() if pd.notna(row.get('title')) else None,
                    hire_date=row.get('hire_date'),
                    employment_status=str(row.get('employment_status', '')).strip() if pd.notna(row.get('employment_status')) else None,
                    industry=str(row.get('industry', '')).strip() if pd.notna(row.get('industry')) else None
                )
                crud.create_employee(db=db, employee=employee_create)
                created_count += 1
        
        return {"created": created_count, "updated": updated_count, "message": "导入完成"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")

@router.post("/batch-link-photos/")
async def batch_link_photos(
    photos: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if not photos:
        raise HTTPException(status_code=400, detail="请上传照片文件")

    all_employees = crud.get_all_employee_names(db)

    matched = []
    unmatched = []
    errors = []

    print(f"[batch-link-photos] received {len(photos)} files")
    for photo in photos:
        fn = photo.filename
        print(f"[batch-link-photos] file: filename={fn!r}, content_type={photo.content_type}")

        if not fn:
            unmatched.append({"filename": fn or "(empty)", "reason": "文件名为空"})
            continue

        filename = fn
        stem = os.path.splitext(filename)[0].strip()

        if not stem:
            unmatched.append({"filename": filename, "reason": "无法解析文件名"})
            continue

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            unmatched.append({"filename": filename, "reason": "不支持的格式"})
            continue

        best_match = None
        best_length = 0
        for emp_id, emp_name in all_employees:
            if emp_name in stem and len(emp_name) > best_length:
                best_match = (emp_id, emp_name)
                best_length = len(emp_name)

        if not best_match:
            unmatched.append({"filename": filename, "reason": f"文件名'{stem}'中未匹配到任何员工姓名，请检查是否与Excel中的姓名一致"})
            continue

        matched_employee_id, matched_name = best_match

        safe_filename = f"{matched_employee_id}.jpg"
        photo_path = os.path.join(PHOTOS_DIR, safe_filename)

        try:
            with open(photo_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

            face_embedding = face_service.extract_embedding(photo_path)

            crud.update_employee_photo(
                db=db,
                employee_id=matched_employee_id,
                photo_path=photo_path,
                face_embedding=face_embedding
            )

            matched.append({
                "filename": filename,
                "name": matched_name,
                "employee_id": matched_employee_id,
                "has_face": bool(face_embedding)
            })
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})

    return {
        "matched": matched,
        "matched_count": len(matched),
        "unmatched": unmatched,
        "unmatched_count": len(unmatched),
        "errors": errors,
        "message": f"成功匹配 {len(matched)} 人，未匹配 {len(unmatched)} 人"
    }

@router.get("/employees-without-photo/")
def get_employees_without_photo(db: Session = Depends(get_db)):
    employees = crud.get_employees_without_photo(db)
    return {
        "count": len(employees),
        "employees": [
            {"employee_id": e.employee_id, "name": e.name}
            for e in employees
        ]
    }