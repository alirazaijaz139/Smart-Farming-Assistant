from typing import Optional
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class FarmCreate(BaseModel):
    name: str
    location: str
    size_acres: Optional[float] = None


class FarmOut(FarmCreate):
    id: int

    class Config:
        from_attributes = True