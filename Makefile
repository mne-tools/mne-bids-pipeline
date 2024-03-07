# simple makefile to simplify repetetive build env management tasks under posix

all: clean

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
	openneuro-py --version
	mri_convert --version
	mne_bids --version
	mne sys_info

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/docs/site/index.html')"
