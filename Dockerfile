FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]