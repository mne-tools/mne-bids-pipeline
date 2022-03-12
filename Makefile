# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CODESPELL_SKIPS ?= "docs/site/*,*.html"
CODESPELL_DIRS ?= scripts/ docs/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf site

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-cache

test: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' mne

install_user:
	$(PYTHON) -m pip install --user --upgrade --progress-bar off -r requirements.txt

install_user_tests:
	$(PYTHON) -m pip install --user --upgrade --progress-bar off -r tests/requirements.txt

install:
	$(PYTHON) -m pip install --upgrade --progress-bar off -r requirements.txt

install_tests:
	$(PYTHON) -m pip install --upgrade --progress-bar off -r tests/requirements.txt

doc:
	./docs/build-docs.sh

check:
	which python
	git-annex version
	datalad --version
	openneuro-py --version
	mri_convert --version
	mne_bids --version
	mne sys_info

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

flake-code:
	flake8 ./run*.py ./scripts --exclude ./scripts/freesurfer/contrib

flake-config:
	flake8 ./config.py --ignore=E501,W503,W504

flake: flake-code flake-config
	@echo "flake8 passed"

codespell:  # running manually
	@codespell --builtin clear,rare,informal,names,usage -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell --builtin clear,rare,informal,names,usage -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne
