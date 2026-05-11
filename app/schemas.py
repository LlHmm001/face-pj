from pydantic import BaseModel
from datetime import date
from typing import Optional

class EmployeeCreate(BaseModel):
    employee_id: str
    name: str
    department: Optional[str] = None
    title: Optional[str] = None
    hire_date: Optional[date] = None
    employment_status: Optional[str] = None
    industry: Optional[str] = None

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    title: Optional[str] = None
    hire_date: Optional[date] = None
    employment_status: Optional[str] = None
    industry: Optional[str] = None

class EmployeeResponse(BaseModel):
    id: int
    employee_id: str
    name: str
    department: Optional[str]
    title: Optional[str]
    hire_date: Optional[date]
    employment_status: Optional[str]
    industry: Optional[str]
    photo_path: Optional[str]

    class Config:
        from_attributes = True

class FaceMatchResult(BaseModel):
    employee_id: str
    name: str
    department: Optional[str]
    title: Optional[str]
    similarity: float
    confidence: Optional[float] = None
    verified: Optional[bool] = None
    photo_path: Optional[str]
    employment_status: Optional[str] = None
    hire_date: Optional[date] = None

class SearchRequest(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    employment_status: Optional[str] = None
    industry: Optional[str] = None

class MultiFaceResult(BaseModel):
    face_index: int
    bbox: dict
    matches: list