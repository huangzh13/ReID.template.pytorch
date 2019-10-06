# ReID.template.pytorch

## Results

| | Rank@1 | mAP| 
| -------- | ----- | ---- | 
| Market1501 | 93.7% | 84.2% |
| DukeMTMC | 86.4% | 74.5% | 
| CUHK03-labeled | 75.9% | 72.5% | 
| CUHK03-detected | 71.2% | 68.0% | 
| CUHK01 | [] | [] | 


## Test

```shell script
python test.py --config ./configs/sample_CUHK03L.yaml
python test.py --config ./configs/sample_CUHK03D.yaml
python test.py --config ./configs/sample_DukeMTMC.yaml
python test.py --config ./configs/sample_Market1501.yaml
```

## Train

```shell script
python train.py --config ./configs/sample_CUHK03L.yaml
python train.py --config ./configs/sample_CUHK03D.yaml
python train.py --config ./configs/sample_DukeMTMC.yaml
python train.py --config ./configs/sample_Market1501.yaml
```
