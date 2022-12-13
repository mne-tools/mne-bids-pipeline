# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CODESPELL_DIRS ?= mne_bids_pipeline/ docs/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf site

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-cache

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

ruff:
	ruff .
	@echo "ruff passed"

codespell:  # running manually; auto-fix spelling mistakes
	@codespell --write-changes $(CODESPELL_DIRS)

codespell-error:  # running on travis; override interactivity seting
	@codespell -i 0 -q 7 $(CODESPELL_DIRS)

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/docs/site/index.html')"
