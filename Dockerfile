# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Using --no-cache-dir reduces image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and models
# This includes the FastAPI app and the ML model files
COPY src/ /app/src/
COPY models/ /app/models/

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use uvicorn to run the FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
