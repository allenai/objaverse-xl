install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy objaverse
	python -m pylint objaverse
	python -m black --check objaverse
	python -m isort --check-only objaverse
	python -m black --check scripts --exclude scripts/rendering/blender-3.2.2-linux-x64/
	python -m isort --check-only scripts/**/*.py --skip scripts/rendering/blender-3.2.2-linux-x64/

format: ## [Local development] Auto-format python code using black, don't include blender
	python -m isort objaverse
	python -m black objaverse
	python -m isort scripts/**/*.py --skip scripts/rendering/blender-3.2.2-linux-x64/
	python -m black scripts --exclude scripts/rendering/blender-3.2.2-linux-x64/

test: ## [Local development] Run unit tests
	JUPYTER_PLATFORM_DIRS=1 python -m pytest -x -s -v tests

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'