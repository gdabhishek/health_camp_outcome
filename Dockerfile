FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update

RUN apt-get update && apt-get install libgomp1 libomp-dev  -y

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]