#!/bin/bash
python3 experiment_scalableGPL.py  --sim sim1 --Ti 50 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 50 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 50 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 50 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 50 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 100 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 100 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 100 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 100 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 100 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 200 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 200 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 200 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 200 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim1 --Ti 200 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 50 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 50 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 50 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 50 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 50 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 100 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 100 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 100 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 100 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 100 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 200 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 200 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 200 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 200 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim2 --Ti 200 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 50 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 50 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 50 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 50 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 50 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 100 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 100 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 100 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 100 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 100 --s 40 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 200 --s 0 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 200 --s 10 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 200 --s 20 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 200 --s 30 --n 10
python3 experiment_scalableGPL.py  --sim sim3 --Ti 200 --s 40 --n 10


#TODO check why matrix inverse is not working


#TODO run sydney house data
try smaller n fr


#signal predictions
bseline: baisan liner regresseion (sklearn)
gp on the data of single node independently - signal task scenario

for task want to model uncertainty --> no deep nets etc
need to print uncertainty
results: how predictions are - do mean
show mean square error and test likelihood - log likelihood on test points, k fold validation (20 times, report avg metric)
leave one node for some time out, train on everythong except from that node in train, predict train for that node --> fold

condition on half, paramtere for testing


15 -- 200
15 experimenets

10 subjects


gamma: justiƒy prior, weight kl terms - justifications
scale down kl gamma 10^-5

email to E to reference KL terms


kak v deep GP delaut normalizazijy?










