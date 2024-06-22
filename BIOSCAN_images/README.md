# BIOSCAN-5M

BIOSCAN-5M Dataset images. 

###### <h3> Image Access

Accessing the dataset images is facilitated by the following directory structure used to organize the dataset images:

```plaintext
bioscan5m/images/[imgtype]/[split]/[chunk]/[{processid}.jpg]
```



###### <h3> BIOSCAN-5M Data Structure
- `[processid]`: A unique identifier given by the collector.
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
  - `val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, `other_heldout` splits: These do not use chunk directories since each split has less than 50k images.

Note that the `val`, `test`, `val_unseen`, `test_unseen`, `key_unseen`, and `other_heldout` splits are all part of the evaluation partition of the `original_256` and `cropped_256` image packages.
