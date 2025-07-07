# based on https://docs.streamlit.io/deploy/tutorials/docker
FROM python:3.11-slim

WORKDIR /app

# copy trialmatcher local package to the container
COPY src/ /app/src
# install trialmatcher
RUN pip install --no-cache-dir /app/src
# install streamlit
RUN pip install --no-cache-dir streamlit

EXPOSE 8501

CMD ["streamlit", "run", "/app/src/trialmatcher/app/trialmatcher_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
