FROM python:3.11

EXPOSE 8080
WORKDIR /text_to_analytics_chat

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "text_to_analytics_chat.py", "--server.port=8080", "--server.address=0.0.0.0"]
