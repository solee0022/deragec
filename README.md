# deragec
The official implementation for the ACL 2025 paper **_ DeRAGEC: Denoising Named Entity Candidates with Synthetic Rationale for ASR Error Correction. _**

## Requirements
* Python 3.10
* PyTorch 2.5.1
* Transformers 4.45.0
* CUDA 12.4
```conda env create --file environment.yaml```

## Inference
### Filterining Named_Entities
To run the filtering process and generate the filtered results in the output folder, execute the following command:
``` python inference_filtering.py```
The filtered output will be saved automatically to the output directory.

### Generative Error Correction (GEC)
To run the GEC module and generate corrected outputs, execute the following command:
``` python inference_gec.py```
The corrected output will be printed directly to the console.
