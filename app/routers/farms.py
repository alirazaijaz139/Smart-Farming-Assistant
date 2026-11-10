from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..security import get_current_user

router = APIRouter(prefix="/farms", tags=["Farms"])


@router.post("/", response_model=schemas.FarmOut, status_code=201)
def create_farm(
    farm: schemas.FarmCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    new_farm = models.Farm(**farm.model_dump(), owner_id=user.id)
    db.add(new_farm)
    db.commit()
    db.refresh(new_farm)
    return new_farm


@router.get("/", response_model=List[schemas.FarmOut])
def my_farms(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    return db.query(models.Farm).filter(models.Farm.owner_id == user.id).all()