# BIOSCAN-5M

Here are the functions handling:

- Download dataset files
- Dataset statistics 
- Data distributions 
- Data split

###### <h3> Dataset Access

The BIOSCAN-5M dataset is available on [GoogleDrive](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0) folder.
To download a file from GoogleDrive run the following:

```bash
python main.py --download --file_to_download <file_name>
``` 

The list of files available for download from GoogleDrive are:

 - Metadata: BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip
 - Bounding Box: BIOSCAN_5M_Insect_bbox.tsv
 - BIOSCAN_5M_original_full.{01:05}.zip
 - BIOSCAN_5M_original_256.zip
    - BIOSCAN_5M_original_256_pretrain.zip
    - BIOSCAN_5M_original_256_train.zip
    - BIOSCAN_5M_original_256_eval.zip
 - BIOSCAN_5M_cropped.{01:02}.zip
 - BIOSCAN_5M_cropped_256.zip
    - BIOSCAN_5M_cropped_256_pretrain.zip
    - BIOSCAN_5M_cropped_256_train.zip
    - BIOSCAN_5M_cropped_256_eval.zip

###### <h3> Dataset Statistics
To see the statistics of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --attr_stat <attribute_types>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.


###### <h3> Dataset Distribution
To see the category distribution of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --attr_dist <attribute_types>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.

###### <h3> Taxonomic Class Distribution
<div align="center">
  <img src="https://github.com/zahrag/BIOSCAN-5M/blob/main/BIOSCAN_images/repo_images/class_order_stats.png" 
       alt="An array of category distribution of the taxonomic level class." />
  <p><b>Figure 2:</b> The category distribution of the taxonomic level class of the BIOSCAN-5M dataset.</p>
</div>

###### <h3> Insect Statistics
To see the statistics of the class Insect of the BIOSCAN-5M dataset, run the following:

```bash
python main.py --level_name class Insecta --attr_stat <attribute_types>
``` 
The attribute type can be <code>genetic</code>, <code>geographic</code>, and <code>size</code>.

###### <h3> Challenges 
The high similarity between samples of different species highlights significant classification challenges. 

<div align="center">
  <img src="https://github.com/zahrag/BIOSCAN-5M/blob/main/BIOSCAN_images/repo_images/species_f.png" 
       alt="Sample images of distinct species from the Order Diptera." />
  <p><b>Figure 2:</b> Sample images of distinct species from the Order Diptera, 
                      which comprises about 50% of our dataset. High similarity between samples of different species 
highlights significant classification challenges.</p>
</div>