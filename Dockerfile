FROM python:3.8-slim-buster

# Update package lists and install necessary CLI tools
RUN apt update -y && apt install -y curl nano bash 

WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your application
CMD ["python", "app.py"]
