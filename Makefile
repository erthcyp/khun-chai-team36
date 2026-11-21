# Single entry point for humans and Codex.
.PHONY: install install-clip dev tcc rsu rsu-clip check

PY ?= python3
export PYTHONPATH := $(CURDIR)

install:
	$(PY) -m pip install -r requirements.txt

install-clip:
	$(PY) -m pip install -r requirements-clip.txt

tcc:
	$(PY) -m uvicorn apps.tcc.main:app --reload --port $${API_PORT:-8000}

rsu:
	$(PY) -m apps.rsu.main --mock --rsu-id $${RSU_ID:-RSU-01}

rsu-clip:
	$(PY) -m apps.rsu.main --images $${IMAGES:-data/samples}

# TCC + two mock RSUs; open http://localhost:8000
dev:
	@trap 'kill 0' INT; \
	$(MAKE) tcc & sleep 2; \
	RSU_ID=RSU-01 $(MAKE) rsu & RSU_ID=RSU-02 $(MAKE) rsu & wait

check:
	$(PY) -m ruff check semcom apps tests
	$(PY) -m pytest -q
