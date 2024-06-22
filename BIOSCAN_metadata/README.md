# BIOSCAN-5M

BIOSCAN-5M metadata file.

###### <h3> Metadata file
Accessing multiple types of the metadata file of the BIOSCAN-5M dataset is facilitated by the following directory structure used to organize the dataset metadata:

```plaintext
bioscan5m/metadata/[type]/BIOSCAN_5M_Insect_Dataset_metadata.[type_extension]
```

- `[type]`: File type of the metadata file:
  - `CSV`
  - `JSON-LD`
- `[type_extension]`:
  - `csv`
  - `jsonld`

###### <h3> Metadata fields

|   | **Field**                    | **Description**                                                       | **Type**   |
|---|------------------------------|-----------------------------------------------------------------------|------------|
| 1 | `processid`                  | A unique identifier given by the collector.                           | String     |
| 2 | `sampleid`                   | A unique identifier given by the collector.                           | String     |
| 3 | `taxon`                      | Bio.info: Most specific taxonomy rank.                                | String     |
| 4 | `phylum`                     | Bio.info: Taxonomic classification label at phylum rank.              | String     |
| 5 | `class`                      | Bio.info: Taxonomic classification label at class rank.               | String     |
| 6 | `order`                      | Bio.info: Taxonomic classification label at order rank.               | String     |
| 7 | `family`                     | Bio.info: Taxonomic classification label at family rank.              | String     |
| 8 | `subfamily`                  | Bio.info: Taxonomic classification label at subfamily rank.           | String     |
| 9 | `genus`                      | Bio.info: Taxonomic classification label at genus rank.               | String     |
| 10| `species`                    | Bio.info: Taxonomic classification label at species rank.             | String     |
| 11| `dna_bin`                    | Bio.info: Barcode Index Number (BIN).                                 | String     |
| 12| `dna_barcode`                | Bio.info: Nucleotide barcode sequence.                                | String     |
| 13| `country`                    | Geo.info: Country associated with the site of collection.             | String     |
| 14| `province_state`             | Geo.info: Province/state associated with the site of collection.      | String     |
| 15| `coord-lat`                  | Geo.info: Latitude (WGS 84; decimal degrees) of the collection site.  | Float      |
| 16| `coord-lon`                  | Geo.info: Longitude (WGS 84; decimal degrees) of the collection site. | Float      |
| 17| `image_measurement_value`    | Size.info: Number of pixels occupied by the organism.                 | Integer    |
| 18| `area_fraction`              | Size.info: Fraction of the original image the cropped image comprises. | Float      |
| 19| `scale_factor`               | Size.info: Ratio of the cropped image to the cropped_256 image.       | Float      |
| 20| `inferred_ranks`             | An integer indicating at which taxonomic ranks the label is inferred. | Integer    |
| 21| `split`                      | Split set (partition) the sample belongs to.                          | String     |
| 22| `index_bioscan_1M_insect`    | An index to locate organism in BIOSCAN-1M Insect metadata.            | Integer    |
| 23| `chunk`                      | The packaging subdirectory name (or empty string) for this image.     | String     |

