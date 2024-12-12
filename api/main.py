from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model_path = "../saved_models/mod_v1.keras"

MODEL = tf.keras.models.load_model(model_path)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def echo():
    return "Pong!!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3000)