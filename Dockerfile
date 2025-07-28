FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (only done during build)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app files
COPY main.py .

# Run your script
CMD ["python", "main.py"]
