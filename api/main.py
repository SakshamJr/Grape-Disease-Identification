from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = pickle.load(
    open(
        "C:/Users/N I T R O V15/OneDrive/Documents/Learning-ML/codebasics/Deep Learning/Grape Disease Identification/model_ver_1.pkl",
        "rb",
    )
)
class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def ping():
    return "Hello, I am server is alive!"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict_disease(file: UploadFile):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])*1
    return {
        'class':predicted_class,
        'confidence':confidence
    }
