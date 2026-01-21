# ==================================================
# Base Image (Python 3.10 - matches local setup)
# ==================================================
FROM python:3.10-slim

# ==================================================
# Environment Variables
# ==================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ==================================================
# System Dependencies
# ==================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==================================================
# Set Working Directory
# ==================================================
WORKDIR /app

# ==================================================
# Copy Requirements & Install Dependencies
# ==================================================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================================================
# Copy Project Files
# ==================================================
COPY app.py .
COPY dataset/ dataset/
COPY Models/ Models/
COPY flight_price_prediction.py .
COPY gender_classification.py .
COPY hotel_recommendation.py .
COPY README.md .

# ==================================================
# Expose Streamlit Port
# ==================================================
EXPOSE 8501

# ==================================================
# Health Check (Optional)
# ==================================================
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ==================================================
# Run Streamlit App
# ==================================================
ENTRYPOINT ["streamlit", "run", "app.py"]
