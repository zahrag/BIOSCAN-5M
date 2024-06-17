from bioscan_datadownload import make_download
from bioscan_statistic import show_statistics
from bioscan_distribution import show_distributions
from config.data_config import set_config


if __name__ == '__main__':

    configs = set_config()
    make_download(configs)
    show_statistics(configs)
    show_distributions(configs)



