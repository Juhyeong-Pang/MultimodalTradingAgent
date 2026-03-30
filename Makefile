ifeq ($(OS),Windows_NT)
    OPEN_CMD = start
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OPEN_CMD = xdg-open
    endif
    ifeq ($(UNAME_S),Darwin)
        OPEN_CMD = open
    endif
endif

URL = http://127.0.0.1:8000/docs

run_server:
	(sleep 5 && $(OPEN_CMD) http://localhost:8000/docs) &
	uvicorn main:app --host 127.0.0.1 --port 8000 --reload

backtest:
	python src/models/backtester.py

build:
	docker build -t mmta .

startApp:
	docker run -p 8000:8000 --env-file .env mmta

launch:
	docker build -t mmta .
	(sleep 15 && $(OPEN_CMD) http://localhost:8000/docs) &
	docker run -p 8000:8000 --env-file .env mmta