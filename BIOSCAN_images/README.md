# BIOSCAN-5M

BIOSCAN-5M Dataset images. 

###### <h3> Image Access
All image packages of the BIOSCAN-5M dataset are available in the [GoogleDrive](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0),
within [BIOSCAN_5M_IMAGES](https://drive.google.com/drive/u/1/folders/1tZ5V_qWSPdDwD90oLz_Uqykp1AoBzLVM) folder.
Accessing the dataset images is facilitated by the following directory structure used to organize the dataset images:

```plaintext
bioscan5m/images/[imgtype]/[split]/[chunk]/[{processid}.jpg]
```



###### <h3> BIOSCAN-5M Data Structure
- `[imgtype]`: Type of the image, which can be one of the following:
  - `original_full`
  - `cropped`
  - `cropped_256`
  - `original_256`
- `[split]`: Data split, which can be one of the following:
  - `pretrain`
  - `train`
  - `val`
  - `test`
  - `val_unseen`
  - `test_unseen`
  - `key_unseen`
  - `other_heldout`
- `[chunk]`: Determined by using the first two or one characters of the MD5 checksum (in hexadecimal) of the `[processid]`. 
  - `pretrain` split: The files are organized into 256 directories using the first two letters of the MD5 checksum of the `[processid]`.
  - `train` and `other_heldout` splits: The files are organized into 16 directories using the first letter of the MD5 checksum.
  - `val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, `other_heldout` splits: These files are organized into chunk directories since each split has less than 50k images.
- `[processid]`: A unique identifier assigned by the collector to each dataset sample.


Note that the `val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, and `other_heldout` splits are all part of the evaluation partition of the `original_256` and `cropped_256` image packages.
