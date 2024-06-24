# BIOSCAN-5M

BIOSCAN-5M crop and resize tools.

###### <h3> Cropping images
To utilize our crop, and/or resize tool please download the cropping checkpoint available in the [GoogleDrive](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0),
within [BIOSCAN_5M_CropTool/checkpoint](https://drive.google.com/drive/u/1/folders/1GiUHLOL-oUr2wBtb58LB0BGv2ymjj2jS) folder.
Subsequently, please visit the [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M), and follow the instructions, 
which facilitate cropping as well as resizing images. 


### Cropping model performance comparison

We compare the performance of the DETR model used for cropping, trained with the extra 837 images (NWC-837) that were previously not well-cropped, to the model used for BIOSCAN-1M. We report the Average Precision (AP) and Average Recall (AR) computed on an additional validation set consisting of 100 images that were not well-cropped previously (NWC-100-VAL), as well as the images (IP-100-VAL + IW-150-VAL) used to evaluate the cropping tool's model used in BIOSCAN-1M. Our updated model performs considerably better on NWC-100-VAL while giving comparable performance on the original validation set of images.

|     |               | NWC-100-VAL   | NWC-100-VAL  | IP-100-VAL + IW-150-VAL  | IP-100-VAL + IW-150-VAL  |
|------------|-----------------------------|---------------|--------------------------|---------------------------------|---------------------------------------|
|       Dataset     |          Training Data                    | AP [0.75]     | AR [0.50:0.95]           |AP [0.75]                        | AR [0.50:0.95]                           |
| BIOSCAN-1M | IP-1000 + IW-1000           | 0.257         | 0.485                    | **0.922**                       | **0.894**                            |
| BIOSCAN-5M | IP-1000 + IW-1000 + NWC-837 | **0.477**     | **0.583**                | 0.890                           | 0.886                                 |





### Bounding box
The bounding box information of BIOSCAN-5M images obtained using our cropping tool is available through the [GoogleDrive](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0),
within [BIOSCAN_5M_CropTool/bounding_box](https://drive.google.com/drive/u/1/folders/1i6mSf5P6nmc228RUOfVwer6TVjZXUzeP) folder.

### <h4> Bounding box file information

|   | **Field**         | **Description**                                                  | **Type** |
|---|-------------------|------------------------------------------------------------------|----------|
| 1 | `processid`       | A sample unique identifier given by the collector.               | String   |
| 2 | `width_original`  | Width of the orignal image.                                      | Integer  |
| 3 | `height_original` | Height of the original image.                                    | Integer  |
| 4 | `x0`              | The x-coordinate of the top-left corner of the bounding box.     | Integer  |
| 5 | `y0`              | The y-coordinate of the top-left corner of the bounding box.     | Integer  |
| 6 | `x1`              | The x-coordinate of the bottom-right corner of the bounding box. | Integer  |
| 7 | `y1`              | The y-coordinate of the bottom-right corner of the bounding box. | Integer  |
| 8 | `scale_factor`    | Ratio of the cropped image to the cropped_256 image.             | Float    |
| 9 | `area_fraction`   | Fraction of the original image the cropped image comprises.      | Float    |


### Size information
Using the bounding box data, we have computed **scale_factor** and **area_fraction** information representing the size of the organism.
The code facilitates these calculations are available in the python script `bioscan_bbox.py`. 
