FROM python:3.14

WORKDIR /
RUN dir 
# Install pipx
RUN pip install poetry

COPY pyproject.toml poetry.lock README.md /

RUN poetry install --no-root


# Install all packages using poetry
EXPOSE 12000

# Finalize startup command: Uvicorn.

CMD [ "uvicorn" , "12000"]