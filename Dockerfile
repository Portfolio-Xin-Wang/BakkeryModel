FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-directory

# Copy application code
COPY ./ ./

EXPOSE 12000

# Start the application using uvicorn directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "12000"]