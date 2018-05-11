.PHONY: all
all:
	pip install -r requirements.txt
	mkdir models
	make dataset


.PHONY: train
train:
	nohup python train.py &


.PHONY: dataset
fetch:
	aws s3 cp s3://nlp-concierge/dataset ./dataset --recursive


.PHONY: up-dataset
upload:
	aws s3 cp ./dataset s3://nlp-concierge/dataset --recursive
