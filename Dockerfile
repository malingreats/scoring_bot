FROM tiangolo/meinheld-gunicorn-flask:python3.7 as base

FROM base
RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY XGBoost.sav /app
COPY ./app /app
