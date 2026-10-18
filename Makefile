PY := .venv/bin/python
.PHONY: setup audit index train predict test
setup:
	uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt --index-strategy unsafe-best-match && VIRTUAL_ENV=.venv uv pip install -e .
audit:
	$(PY) -m termmap.cli audit
index:
	$(PY) -m termmap.cli build-index
train:
	$(PY) -m termmap.cli train
predict:
	$(PY) -m termmap.cli predict --input $(INPUT) --output $(OUTPUT)
test:
	$(PY) -m pytest -q
