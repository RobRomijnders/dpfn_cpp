.PHONY: build

build:
	rm -rf build/
	rm -f *linux-gnu.so
	pip install -e .

test:
	nose2 -v

hint:
	pytype tests

test-hint: test hint
	echo 'Finished running tests and checking type hints'

lint:
	cpplint src/*.cpp
	cpplint src/*.hpp
	pylint tests
	pycodestyle tests/*.py
	# pycodestyle tests/**/*.py
	pydocstyle tests/*.py --ignore=D103,D104,D107,D203,D204,D213,D215,D400,D401,D404,D406,D407,D408,D409,D413
	# pydocstyle tests/**/*.py --ignore=D103,D104,D107,D203,D204,D213,D215,D400,D401,D404,D406,D407,D408,D409,D413
