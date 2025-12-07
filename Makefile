.PHONY: clean build

build:
	unzip -q data/data.zip -d data/ -x "__MACOSX/*"
	mkdir -p data/load_data/
	mkdir -p data/train_model/

clean:
	rm -rf data/load_data/
	mkdir -p data/load_data/
	rm -rf data/train_model/
	mkdir -p data/train_model/
	rm -rf src/__pycache__/
