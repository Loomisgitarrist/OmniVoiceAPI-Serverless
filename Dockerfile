FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# Install git + ffmpeg (gotcha #1 from RunPod AI)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the omnivoice package locally (vendored, no auth needed)
COPY omnivoice /app/omnivoice
ENV PYTHONPATH=/app:$PYTHONPATH

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
