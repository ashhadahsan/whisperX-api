# whisperX-api

## Requirements
Before running the server, please make sure that you have installed Python and its required packages. You can install the required packages by running the command:

pip install -r requirements.txt


## Running the server
To start the server, run the command:
uvicorn app:app --reload


This will start the server using the `app` object defined in the `app.py` file. The `--reload` option is used to enable hot-reloading, which means that the server will automatically restart when changes are made to the code.
Check notebook.ipynb and test.json to see the code to send to the API.

## Docker
A Dockerfile is provided in case you want to build a Docker image of the app. To build the Docker image, run the command:

docker build -t whisperx-api .


This will build a Docker image with the tag `whisperx-api`. You can then start a container from this image using the command:
docker run -p 8000:8000 whisperx-api

This will start a container that listens on port `8000` and forwards requests to the server running inside the container.
