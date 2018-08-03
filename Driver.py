import numpy as np
import os
from random import sample
from Anomaly_NN import train_NN 
anomaly_vid =["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting","RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]
normal_vid = ["Normal_Videos_event","Testing_Normal_Videos_Anomaly","Training_Normal_Videos_Anaomaly"]
ano_data = []
normal_data = []
train_ano = []
train_norm = []
test_ano = []
test_norm = []
ano_width = []
norm_width = []


def train_model():
	train_NN(train_ano,train_norm,test_ano,test_norm,ano_width[125:],norm_width[125:])
	
for folder in anomaly_vid:
	path = "./FOutput/" + folder
	files = os.listdir(path)
	if len(files) == 0:
		continue
	for fl in files:
		with open(path +"/"+fl,'r') as f:
			ano_width.append(float(f.readline().split("\n")[0]))
			for cnt,line in enumerate(f):
				if cnt == 32:
					break
				line = line.split("\n")[0]
				ano_data.append(map(float,line.split(",")))
for folder in normal_vid:
	path = "./FOutput/" + folder
	files = os.listdir(path)
	if len(files) == 0:
		continue
	for fl in files:
		with open(path +"/"+fl,'r') as f:
			norm_width.append(float(f.readline().split("\n")[0]))
			for cnt,line in enumerate(f):
				if cnt == 32:
					break
				line = line.split("\n")[0]
				normal_data.append(map(float,line.split(",")))

train_ano = ano_data[0:4000]
train_norm = normal_data[0:4000]
test_ano = ano_data[4000:]
test_norm = normal_data[4000:]
train_model()
