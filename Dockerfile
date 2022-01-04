FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app
RUN pip install pandas
RUN pip install sqlalchemy
RUN pip install psycopg2
RUN pip install datetime
RUN pip install yfinance
RUN pip install pandas_datareader
