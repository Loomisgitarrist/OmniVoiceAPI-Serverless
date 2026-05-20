FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# Install git + ffmpeg (gotcha #1 from RunPod AI)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the omnivoice package locally (vendored, no auth needed)
COPY omnivoice /app/omnivoice
ENV PYTHONPATH=/app:$PYTHONPATH

# Pre-download the model so cold-start doesn't timeout (build 26190129042 gotcha)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('k2-fsa/OmniVoice')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
