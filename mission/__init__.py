from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
router = APIRouter()

app.mount("/static", StaticFiles(directory="./mission/static"), name="static")
templates = Jinja2Templates(directory="./mission/templates")

from mission import main

