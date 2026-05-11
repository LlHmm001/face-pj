from sqlalchemy import Column, Integer, String, Date, Boolean, Text
from .database import Base

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    department = Column(String, index=True)
    title = Column(String, index=True)
    hire_date = Column(Date)
    employment_status = Column(String, index=True)
    industry = Column(String, index=True)
    photo_path = Column(String)
    face_embedding = Column(Text)