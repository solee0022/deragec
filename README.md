# deragec
🌟 The official implementation for the ACL 2025 paper **_DeRAGEC: Denoising Named Entity Candidates with Synthetic Rationale for ASR Error Correction._** 🌟

## Requirements
* Python 3.10
* PyTorch 2.5.1
* Transformers 4.45.0
* CUDA 12.4
```
conda env create --file environment.yaml
```

## Dataset
We provide one **Common Voice** (cv) datapoint and one **STOP** (stop) datapoint as example inputs for inference. You can find them in the ```examples``` directory.

## Inference
#### Filterining Named_Entities
To run the filtering process and generate the filtered results in the output folder, execute the following command:
``` 
python inference_filtering.py --method DeRAGEC --data_type cv
```

The filtered output will be saved automatically to the ```output``` directory.

#### Generative Error Correction (GEC)
To run the GEC module and generate corrected outputs, execute the following command:
``` 
python inference_gec.py --method DeRAGEC --data_type cv
```
The corrected output will be printed directly to the console.
