FROM python:3.9-slim

# Install system dependencies for audio processing (ffmpeg is required for MP3)
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]