#!/bin/bash

for index in {0..470} 
do 
	python train.py --index ${index}
done
