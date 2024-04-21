
from bioscan_statistic import show_statistics
from bioscan_distribution import show_distributions
from bioscan_split_statistic import show_split_statistics
from bioscan_configuration import set_config


if __name__ == '__main__':

    # ################################# DATASET & MODEL CONFIGURATIONS ##########################################
    configs = set_config()

    # ################################# GET DATASET STATISTICS #############################################
    show_statistics(configs)

    # ################################# GET DATASET DISTRIBUTIONS ##########################################
    show_distributions(configs)

    # ################################# GET SPLIT DISTRIBUTIONS ##########################################
    show_split_statistics(configs)



