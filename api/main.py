from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model_path = "../saved_models/1"

# Load the model using TensorFlow's SavedModel format
MODEL = tf.saved_model.load(model_path)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def echo():
#     return "Pong!!"

@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Disease API"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    # Read the image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    # Use the model to make predictions
    infer = MODEL.signatures['serving_default']  # Default signature
    predictions = infer(tf.convert_to_tensor(img_batch, dtype=tf.float32))['output_0']  # Use correct output key

    predicted_class = CLASS_NAMES[np.argmax(predictions[0].numpy())]
    confidence = np.max(predictions[0].numpy())

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3000)
