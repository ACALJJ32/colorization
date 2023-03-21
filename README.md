## Temporary repository
## Requirements

Please install the dependencies according to ```environment.yml```.

## Usage

Clone the repository
```
git clone https://github.com/ACALJJ32/colorization.git
```

Download pretrained models in root folder and unzip.
```
Baidu Disk: https://pan.baidu.com/s/1hKF8UxiVq3P92qOoDkFQvA
code: 4vyd
```


### Train 
```
  TODO
```

### Test on test set clips
```
# FID, Track1
python VP_code/fid_test_inference.py

# CDC, Track2
python VP_code/cdc_test_inference.py
```

### runtime per frame
```
python VP_code/runtime.py
```


The restored results could be found in ```./output_dir``` folder.

### Test results
```
# FID and CDC results on Baidu Disk
https://pan.baidu.com/s/1lk0kt-iEmYBmWYGo_ER4Ig

code: 5pyv
```