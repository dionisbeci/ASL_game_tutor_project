# 1. Use Python 3.11 (Fixes MediaPipe issues)
FROM python:3.11

# 2. Set working directory
WORKDIR /code

# 3. Install Linux Graphics Drivers (Fixes OpenCV/GL errors)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy specific requirements first (for caching)
COPY requirements.txt .

# 5. Install Python Dependencies
# We use the standard pip, no fancy flags needed
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the app code
COPY . .

# 7. Create a user to run the app (Security requirement for HF)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 8. Run the app on port 7860 (Hugging Face default)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]