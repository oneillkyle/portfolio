from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np

# from elasticapm.contrib.starlette import ElasticAPM, make_apm_client

# try:
#   apm = make_apm_client({
#       'SERVICE_NAME': 'my_python_service',
#       'SECRET_TOKEN': 'supersecrettoken',
#       # SERVER_URL must be set to "fleet-server" if running as a docker container.
#       # if running as a local python script, then set the url to "LOCALHOST"
#       'SERVER_URL': 'http://fleet-server:8200',
#       'ENVIRONMENT': 'development'
#   })
# except Exception as e:
#   print('failed to create client')

app = FastAPI()

# Request model for input data
# class InputData(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# # Load the trained TensorFlow model
# loaded_model = tf.keras.models.load_model('saved_model/iris_model')

# # API endpoint for model inference
# @app.post("/predict")
# def predict(input_data: InputData):
#     # Convert input features to a numpy array
#     features = np.array([
#         [input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]
#     ])

#     # Perform model inference
#     predictions = loaded_model.predict(features)
#     predicted_class = np.argmax(predictions, axis=1)[0]

#     # Map predicted class index to class label
#     class_labels = ["Setosa", "Versicolor", "Virginica"]
#     predicted_label = class_labels[predicted_class]

#     # Convert predicted class to Python integer
#     predicted_class = int(predicted_class)

#     # Create the response JSON
#     response_data = {"predicted_class": predicted_class, "predicted_label": predicted_label}

#     return JSONResponse(content=response_data)

# try:
#   app.add_middleware(ElasticAPM, client=apm)
# except Exception as e:
#   print('failed to add APM Middleware')


@app.get("/test/{message}")
async def custom_message(message: str):
    return {"message": f"Custom Message:  {message}"}


# @app.get("/error")
# async def throw_error():
#     try:
#         1 / 0
#     except Exception as e:
#         apm.capture_exception()
#     return {"message": "Failed Successfully :)"}

# try:
#   apm.capture_message('App Loaded, Hello World!')
# except Exception as e:
#   print('error: ' + e)

if __name__ == '__main__':
    print('Please start the app with the "uvicorn" command as shown in the start.sh script')