from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from fastapi import FastAPI

try:
  apm = make_apm_client({
      'SERVICE_NAME': 'my_python_service',
      'SECRET_TOKEN': 'supersecrettoken',
      # SERVER_URL must be set to "fleet-server" if running as a docker container.
      # if running as a local python script, then set the url to "LOCALHOST"
      'SERVER_URL': 'http://fleet-server:8200',
      'ENVIRONMENT': 'development'
  })
except Exception as e:
  print('failed to create client')

app = FastAPI()

try:
  app.add_middleware(ElasticAPM, client=apm)
except Exception as e:
  print('failed to add APM Middleware')


@app.get("/custom_message/{message}")
async def custom_message(message: str):
    apm.capture_message(f"Custom Message: {message}")
    return {"message": f"Custom Message:  {message}"}


@app.get("/error")
async def throw_error():
    try:
        1 / 0
    except Exception as e:
        apm.capture_exception()
    return {"message": "Failed Successfully :)"}

try:
  apm.capture_message('App Loaded, Hello World!')
except Exception as e:
  print('error: ' + e)

if __name__ == '__main__':
    print('Please start the app with the "uvicorn" command as shown in the start.sh script')