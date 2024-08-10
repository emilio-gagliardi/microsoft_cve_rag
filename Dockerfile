# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure the necessary directories exist
RUN mkdir -p data documents notebooks

# Expose the port that the app runs on
EXPOSE 8000

# Set the environment variable for the environment file
ENV ENV_FILE=.env.docker

# Command to run the application
CMD ["python", "path/to/main.py"]