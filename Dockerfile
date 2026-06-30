# Step 1: Use an official, lightweight Python runtime
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install essential system dependencies for building ChromaDB components
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements first to use Docker caching
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Document the port the app listens on
EXPOSE 8000

# Step 8: Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]