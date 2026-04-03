FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir unsloth

COPY main.py .

ENV TRAINING_BACKEND=unsloth
ENV TRAINING_PORT=8000

EXPOSE 8000
CMD ["python", "main.py"]
