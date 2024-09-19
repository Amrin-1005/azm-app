# # Use a base image
# FROM python:3.12

# # Set the working directory in the container
# WORKDIR /app

# COPY requirements.txt /app/

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the current directory contents into the container
# COPY . /app

# # Install system dependencies for OpenCV and ODBC
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     unixodbc \
#     unixodbc-dev



# # Expose port 5000 for Flask
# EXPOSE 5000

# # Set environment variable
# ENV FLASK_APP=app.py

# # Command to run the Flask app
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]



# Use a slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies early to leverage Docker caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies if needed (e.g., for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unixodbc \
    unixodbc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files to the image
COPY . /app

# Expose the port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Start the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

