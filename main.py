
from BIOSCAN_DATASET.bioscan_datadownload import make_download
from BIOSCAN_DATASET.bioscan_statistic import show_statistics
from BIOSCAN_DATASET.bioscan_distribution import show_distributions
from BIOSCAN_DATASET.bioscan_split import make_split
from BIOSCAN_DATASET.bioscan_split_statistic import show_split_statistics
from BIOSCAN_DATASET.config.data_config import set_config

from BIOSCAN_CLIP.scripts.train_cl import train_bioscan_clip
from BIOSCAN_CLIP.scripts.inference_and_eval import test_bioscan_clip


if __name__ == '__main__':

    configs = set_config()
    make_download(configs)

    show_statistics(configs)
    show_distributions(configs)

    make_split(configs)
    show_split_statistics(configs)

    train_bioscan_clip(configs)
    test_bioscan_clip(configs)




