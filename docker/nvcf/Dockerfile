# Testing Docker container: replay the request + asset retrieval
from python:3.10

RUN python3 -m pip install fire flask

ADD server.py /

CMD ["python3", "/server.py", "run"]