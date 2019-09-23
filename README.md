# ReID.template.pytorch

## Results-baseline

| | Rank@1 | mAP| 
| -------- | ----- | ---- | 
| Market1501 | 93.7% | 84.2% |
| DukeMTMC | 86.4% | 74.5% | 
| CUHK03-labeled | 75.9% | 72.5% | 
| CUHK03-detected | 71.2% | 68.0% | 
| CUHK01 | [] | [] | 

**Results with Flip operating**

| | Rank@1 | mAP| 
| -------- | ----- | ---- | 
| Market1501 | 93.7% | 84.8% |
| DukeMTMC | 87.2% | 75.4% | 
| CUHK03-labeled | 76.6% | 73.5% | 
| CUHK03-detected | 72.6% | 69.2% | 
| CUHK01 | [] | [] | 

### Test

```shell script
python test.py --config ./configs/sample_CUHK03L.yaml
python test.py --config ./configs/sample_CUHK03D.yaml
python test.py --config ./configs/sample_DukeMTMC.yaml
python test.py --config ./configs/sample_Market1501.yaml
```

### Train

```shell script
python train.py --config ./configs/sample_CUHK03L.yaml
python train.py --config ./configs/sample_CUHK03D.yaml
python train.py --config ./configs/sample_DukeMTMC.yaml
python train.py --config ./configs/sample_Market1501.yaml
```

## Results-DSA

| | Rank@1 | mAP| 
| -------- | ----- | ---- | 
| Market1501 | [] | [] |
| DukeMTMC | [] | [] | 
| CUHK03-labeled(MF-Stream) | 75.9% | 73.3% | 
| CUHK03-labeled | [] | [] | 
| CUHK03-detected | [] | [] | 
| CUHK01 | [] | [] | 

**Results with Flip operating**

| | Rank@1 | mAP| 
| -------- | ----- | ---- | 
| Market1501 | [] | [] |
| DukeMTMC | [] | [] | 
| CUHK03-labeled(MF-Stream) | 77.2% | 74.3% | 
| CUHK03-labeled | [] | [] | 
| CUHK03-detected | [] | [] | 
| CUHK01 | [] | [] | 

### Test

```shell script
python test_dsa.py --config ./configs/DSA_CUHK03L.yaml
python test_dsa.py --config ./configs/MF_CUHK03L.yaml
```

### Train

```shell script
python train_dsa.py --config ./configs/DSA_CUHK03L.yaml
python train_dsa.py --config ./configs/MF_CUHK03L.yaml
```