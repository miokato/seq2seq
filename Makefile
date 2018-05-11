.PHONY: all
all:
	pip install -r requirements.txt
	mkdir models
	make dataset


.PHONY: train
train:
	nohup python train.py &


.PHONY: dataset
dataset:
	aws s3 cp s3://nlp-concierge/dataset ./dataset --recursive
