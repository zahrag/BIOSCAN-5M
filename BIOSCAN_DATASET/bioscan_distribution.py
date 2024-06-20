import math
from bioscan_statistic import BioScan
from tabulate import tabulate


class BioScanDists:

    def print_table(self, data_dict, title, print_table=False):

        if not print_table:
            return

        print("\n\n" + "+" + "-" * 143 + "+")
        print(f"\t\t\t\t\t{title}")
        headings = list(data_dict.keys())
        values = list(data_dict.values())
        rows = zip(*values)
        formatted_rows = [
            [f'{val: .2f}' if isinstance(val, float) else str(val) for val in row]
            for row in rows
        ]
        print(tabulate(formatted_rows, headers=headings, tablefmt="grid"))

    def get_dis_dict(self, dataset, attr_list, dict_type='Genetic Attributes'):

        g_dict = {dict_type: [], 'Categories': [],
                  'Most Populated': [], 'Most Populated Size': [],
                  'Least Populated': [], 'Least Populated Size': [],
                  'Mean': [], 'Median': [], 'STD': []}

        for attr in attr_list:

            lst = dataset.data_list_mapping.get(attr)
            attr_dict = dataset.make_data_dict(lst)
            cats = list(attr_dict.keys())

            if 'no_data' in cats:
                del attr_dict['no_data']
                cats.remove('no_data')

            len_cats = sorted([len(attr_dict[c]) for c in cats])
            mean = sum(len_cats)/len(cats)
            n = len(len_cats)
            median = (len_cats[n//2 - 1] + len_cats[n//2]) / 2 if n % 2 == 0 else len_cats[n//2]

            variance = sum((x - mean) ** 2 for x in len_cats) / len(len_cats)
            std = variance ** 0.5

            g_dict[dict_type].append(attr)
            g_dict['Categories'].append(len(cats))
            g_dict['Most Populated'].append(cats[0])
            g_dict['Most Populated Size'].append(len(attr_dict[cats[0]]))
            g_dict['Least Populated'].append(cats[-1])
            g_dict['Least Populated Size'].append(len(attr_dict[cats[-1]]))
            g_dict['Mean'].append(mean)
            g_dict['Median'].append(median)
            g_dict['STD'].append(std)

        return g_dict

    def get_dataset_distribution(self, configs, dataset):

        if configs['attr_dist'] not in ['genetic', 'geographic', 'size']:
            return

        get_attr = configs['attr_dist']

        if get_attr == 'genetic':
            list_attr = list({**dataset.taxonomy_groups_list_dict}.keys())
            list_attr.append('dna_bin')
        elif get_attr == 'geographic':
            list_attr = list({**dataset.geographic_list_dict}.keys())
        elif get_attr == 'size':
            list_attr = list({**dataset.size_list_dict}.keys())
        else:
            raise ValueError('Statistic type is NOT available!')

        g_dict = self.get_dis_dict(dataset, list_attr, dict_type=f'{get_attr.capitalize()} Class Distribution')
        # Print category distribution
        scale = int(math.floor(math.log10(abs(len(dataset.index)))))
        s = 'k' if scale < 6 else 'M'
        n = round(len(dataset.index) / 10 ** scale)
        title = f"{get_attr.capitalize()} Category Distribution of the BIOSCAN-{n}{s} Insect dataset "
        title += f"having {len(dataset.index)} specimens"
        self.print_table(g_dict, title, print_table=True)


def show_distributions(configs):
    """
    This function shows/saves dataset distributions.
    :param configs: Configurations.
    :return:
    """

    if not configs['attr_dist']:
        return

    dataset = BioScan()
    dataset.get_statistics(configs["metadata"], level_name=configs["level_name"])

    vis_dists = BioScanDists()
    vis_dists.get_dataset_distribution(configs, dataset)
