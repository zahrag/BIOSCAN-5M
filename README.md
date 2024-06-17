# BIOSCAN-5M



###### <h3> Overview
This repository contains the code and data related to the to the [BIOSCAN-5M ](https://biodiversitygenomics.net/5M-insects/).  
BIOSCAN-5M is a comprehensive dataset comprising multi-modal information about over 5 million specimens, with 98% being insects. 

Anyone interested in using BIOSCAN-1M Insect dataset and/or the corresponding code repository, 
please cite the [Paper]():

```
@inproceedings{gharaee24,
    title={BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity},
    author={Gharaee, Z. and Lowe, S. C. and Gong, Z. and Arias. P. M. and Wang, A. T. and Pellegrino, N. 
    and Haurum, J. B. and Zarubiieva, I. and Kari, L. and Steinke, D. and Taylor, G. W. and Fieguth, P. and Chang, A. X.},
    publisher={arxiv},
    year={2024},
}
```
###### <h3> Dataset Access
The dataset image packages and metadata file are accessible for download through 
the [GoogleDrive folder](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0).

###### <h3> Dataset
We present BIOSCAN-5M dataset to the machine learning community with valuable information about insect's biodiversity. 
Each record of the BIOSCAN-5M dataset contains six primary attributes:
* DNA Barcode Sequence
* Barcode Index Number (BIN)
* Biological Taxonomy Classification
* RGB image
* Geographical information 
* Size information


###### <h3> Image Packages
The BIOSCAN-5M dataset comprises resized and cropped images.
We have provided various packages of the BIOSCAN-5M dataset, each tailored for specific purposes.

* BIOSCAN_original.zip: The raw images of the dataset.
* BIOSCAN_cropped.zip: Images after cropping with our cropping tool.
* BIOSCAN_original_256.zip: Original images resized to 256 on their shorter side.
  * BIOSCAN_original_256_pretrain.zip
  * BIOSCAN_original_256_train.zip
  * BIOSCAN_original_256_eval.zip
* BIOSCAN_cropped_256.zip: Cropped images resized to 256 on their shorter side.
  * BIOSCAN_cropped_256_pretrain.zip
  * BIOSCAN_cropped_256_train.zip
  * BIOSCAN_cropped_256_eval.zip

###### <h3> Metadata 
The dataset metadata file **BIOSCAN_5M_Insect_Dataset_metadata** contains biological information, geographic information as well as 
size information of the organisms. We created both CSV and JSONLD types of the metadata file.


 

