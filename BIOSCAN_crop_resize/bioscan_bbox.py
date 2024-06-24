import pandas as pd
import dataset_helper


def get_round(val, n=2):
    return round(val, n)


def _area_fraction(w_bbx, h_bbx, w, h):

    a_factor = (w_bbx * h_bbx) / (w * h)
    a_factor = get_round(a_factor, 4)
    return a_factor


def _scale_factor(w_bbx, h_bbx, resized_dim=256):

    s_factor = (pd.DataFrame({'w': w_bbx, 'h': h_bbx}).min(axis=1)) / resized_dim
    # s_factor = (w_bbx.combine(h_bbx, min)) / resized_dim

    s_factor = get_round(s_factor, 4)
    return s_factor


def get_size_bbx(configs):
    """
    This function computes the size data (area_fraction and scale_factor) from the bounding box information
    detected by our crop-tool.
    :param configs: Configurations.
    :return:
    """

    if not configs['attr_stat']:
        return

    bbx = dataset_helper.read_csv(configs['bbox'])

    w_bbx = bbx['x1'] - bbx['x0']
    h_bbx = bbx['y1'] - bbx['y0']

    area_frac = _area_fraction(w_bbx, h_bbx, bbx['width_original'], bbx['height_original'])
    scale_fact = _scale_factor(w_bbx, h_bbx, resized_dim=256)

    return area_frac, scale_fact
