## Adversarial Speech Commands
Fork of original work by github user `nesl` being used to validate and expand upon results.


### Requirements
Python 3.9 (3.9.7 used during our recent work)
Tensorflow 2.6 (2.6.0 used during our recent work, original implementation version undocumented)

### Setup

Download datset
```
bash download_dataset.sh
```

Download model:
```
bash download_checkpoint.sh
```
### Attack Experiment
To run:
```
./run_attack.sh dataset_dir ckpts_dir limit max_iters test_size
```
