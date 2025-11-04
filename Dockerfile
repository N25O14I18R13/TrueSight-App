FROM python:3.9

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

RUN pip uninstall -y opencv-python opencv-python-headless

RUN pip install opencv-python-headless

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 7860

CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]