FROM python:3.12-slim

WORKDIR /app

# Install the libraries first (this step is reused if requirements.txt does not change)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

# Copy only what the dashboard needs
COPY app.py .
COPY src/ src/
COPY models/ models/
COPY data/processed/movies_financial.csv data/processed/movies_financial.csv

EXPOSE 8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]