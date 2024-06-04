from pydantic import BaseModel
from typing import Optional

class VideoResponse(BaseModel):
    id: int
    url: Optional[str]
    title: str
    ft_like: int
    name: str
    img: str
    liked: bool
    author_id: int

    class Config:
        orm_mode = True
