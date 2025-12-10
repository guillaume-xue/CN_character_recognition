.PHONY: clean build

build:
	unzip -q data/data.zip -d data/ -x "__MACOSX/*"

clean:
	rm -rf data/load_data/
	rm -rf data/train_model/
	rm -rf src/__pycache__/
