from sqlalchemy.orm import Session
from datetime import date
from .models import Employee
from .schemas import EmployeeCreate, EmployeeUpdate

def get_employee(db: Session, employee_id: str):
    return db.query(Employee).filter(Employee.employee_id == employee_id).first()

def get_employee_by_id(db: Session, id: int):
    return db.query(Employee).filter(Employee.id == id).first()

def get_employees(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Employee).offset(skip).limit(limit).all()

def search_employees(db: Session, name: str = None, department: str = None, 
                    employment_status: str = None, industry: str = None,
                    hire_date_from: date = None, hire_date_to: date = None):
    query = db.query(Employee)
    
    if name:
        query = query.filter(Employee.name.ilike(f"%{name}%"))
    if department:
        query = query.filter(Employee.department.ilike(f"%{department}%"))
    if employment_status:
        query = query.filter(Employee.employment_status.ilike(f"%{employment_status}%"))
    if industry:
        query = query.filter(Employee.industry.ilike(f"%{industry}%"))
    if hire_date_from:
        query = query.filter(Employee.hire_date >= hire_date_from)
    if hire_date_to:
        query = query.filter(Employee.hire_date <= hire_date_to)
    
    return query.all()

def create_employee(db: Session, employee: EmployeeCreate, photo_path: str = None, face_embedding: str = None):
    db_employee = Employee(
        employee_id=employee.employee_id,
        name=employee.name,
        department=employee.department,
        title=employee.title,
        hire_date=employee.hire_date,
        employment_status=employee.employment_status,
        industry=employee.industry,
        photo_path=photo_path,
        face_embedding=face_embedding
    )
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee

def update_employee(db: Session, db_employee: Employee, employee_update: EmployeeUpdate):
    if employee_update.name is not None:
        db_employee.name = employee_update.name
    if employee_update.department is not None:
        db_employee.department = employee_update.department
    if employee_update.title is not None:
        db_employee.title = employee_update.title
    if employee_update.hire_date is not None:
        db_employee.hire_date = employee_update.hire_date
    if employee_update.employment_status is not None:
        db_employee.employment_status = employee_update.employment_status
    if employee_update.industry is not None:
        db_employee.industry = employee_update.industry
    
    db.commit()
    db.refresh(db_employee)
    return db_employee

def delete_employee(db: Session, employee_id: str):
    db_employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if db_employee:
        db.delete(db_employee)
        db.commit()
        return True
    return False

def get_all_employees_with_embedding(db: Session):
    return db.query(Employee).filter(Employee.face_embedding.isnot(None)).all()

def get_employee_by_name(db: Session, name: str):
    return db.query(Employee).filter(Employee.name == name).first()

def get_all_employee_names(db: Session):
    return db.query(Employee.employee_id, Employee.name).all()

def get_employees_without_photo(db: Session):
    return db.query(Employee).filter(
        (Employee.photo_path.is_(None)) | (Employee.face_embedding.is_(None))
    ).all()

def update_employee_photo(db: Session, employee_id: str, photo_path: str, face_embedding: str = None):
    db_employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if db_employee:
        db_employee.photo_path = photo_path
        db_employee.face_embedding = face_embedding
        db.commit()
        db.refresh(db_employee)
        return db_employee
    return None