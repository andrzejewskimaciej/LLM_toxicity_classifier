# lightweight python
FROM python:3.10-slim

# system utils
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy into container
COPY ./app /app

# ports
EXPOSE 8000 8501

CMD ["python", "--version"]