.PHONY: install lint typecheck test check demo results assets explorer notebook clean

install:        ## editable install with dev tools
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

typecheck:
	python -m mypy

test:
	python -m pytest --cov --cov-report=term-missing

check: lint typecheck test   ## everything CI runs

demo:           ## synthetic end-to-end run
	vvi demo

results:        ## benchmark + power study -> docs/synthetic (slow, synthetic)
	python scripts/benchmark.py
	python scripts/power_study.py

assets:         ## re-render docs/img (light + dark, PNG + GIF) from synthetic data
	python scripts/render_assets.py

explorer:       ## build the explorer site from synthetic data and serve it
	python scripts/build_site.py build/site
	python -m http.server -d build/site 8000

notebook:       ## re-execute examples/quickstart.ipynb
	python scripts/make_notebook.py

clean:
	rm -rf build .pytest_cache .mypy_cache .ruff_cache .coverage
