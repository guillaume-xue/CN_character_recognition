.PHONY: clean build

build:
	unzip -q data/data.zip -d data/ -x "__MACOSX/*"

clean:
	rm -rf data/preload
	mkdir -p data/preload
	rm -rf src/__pycache__/
