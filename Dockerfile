FROM apache/airflow:3.1.5-python3.11
# Ставим необходимые библиотеки, также был выбран образ с python3.11
# Так как это последняя версия питона, которую подерживает pycaret
RUN pip install --no-cache-dir mlflow=="2.16.0" pycaret
USER root
# Установка необходимой библиотеки для pycaret
RUN apt-get update \
    && apt-get install -y libgomp1 \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow