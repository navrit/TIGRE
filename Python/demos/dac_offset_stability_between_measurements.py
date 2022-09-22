import numpy as np
import matplotlib.pyplot as plt
import os
import re
from natsort import natsorted
from tqdm import tqdm, trange

def check_all_dac_values_files_exist(prefix, folders, filename):
    for f in folders:
        file_path = os.path.join(prefix, f, 'fit_poly_all_datasets', filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
    return

def load_all_dac_values_files(prefix, folders, filename):
    all_dac_values_all_datasets = list()
    for f in folders:
        file_path = os.path.join(prefix, f, 'fit_poly_all_datasets', filename)
        all_dac_values_all_datasets.append(np.load(file_path))
    return all_dac_values_all_datasets

def get_folder_names_of_datasets(prefix, folders):
    regex = r'\d...\d'
    folders_dict = dict()
    for f in folders:
        dataset_path = os.path.join(prefix, f)
        dataset_paths = natsorted(os.listdir(dataset_path))
        tiff_folders = list()
        for d in dataset_paths:
            tiff_folder = os.path.basename(os.path.normpath(d))
            if re.match(regex, tiff_folder):
                tiff_folders.append(tiff_folder)
        folders_dict[f] = tiff_folders
    return folders_dict

def print_stats(statistics, folder_name):
        for l in statistics:
            if l[0] == folder_name:
                print(l)
        print('-------')

def main():
    prefix = os.path.join('f:\\', 'jasper', 'data')
    folders = [
        '20220909_Legoman_nomotion',
        '20220907_Legoman_oldplate',
        '20220905_Legoman_oldplate',
        '20220812_BreastTissueFFPE',
        '20220810_HamNCheese',

        '20220822_ffpe_WhateverBreast',
        '20220819_ffpe_WhateverBreast',
        '20220818_ffpe_WhateverBreast',

        # '20220804_HamnCheeseseries_M'
    ]
    filename = r'all_dac_values.npy'

    check_all_dac_values_files_exist(prefix, folders, filename)
    all_dac_values_all_datasets = load_all_dac_values_files(prefix, folders, filename)

    print(f'Using {len(all_dac_values_all_datasets)} datasets: {folders}\n')

    folders_dict = get_folder_names_of_datasets(prefix, folders)
    for k, v in folders_dict.items():
        print(k, v)
        # for folder in v:
        #     print(os.path.join(prefix, k, folder, f'Scan_{folder}_keV.json'))
        # print('--------')
    print('\n')

    output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dac_offset_stability_between_measurements')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    statistics = list()
    for i in trange(len(all_dac_values_all_datasets), position=0, leave=True):
        data_folders = folders_dict[folders[i]]
        for idx, j in tqdm(enumerate(data_folders), position=1, leave=False):
            d = all_dac_values_all_datasets[i][idx,:,:]
            to_append = [j, list(folders_dict)[i], np.nanmin(d), np.nanmean(d), np.nanmedian(d), np.nanmax(d), np.nanstd(d)]
            statistics.append(to_append)
            d_flat = d.flatten()

            plt.title(str(list(folders_dict)[i] + ' --> ' + j))
            plt.yscale('log')
            plt.xlim(0, 120)
            plt.ylim(1, 1e5)
            plt.hist(d_flat, bins=200, label=' '.join(str(e)+'\n' for e in to_append))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, '{:s}_{:s}.png'.format(j, list(folders_dict)[i])) )
            plt.close('all')


            mean = np.nanmean(d_flat)
            std = np.nanstd(d_flat)
            n = 3
            plt.title(str(list(folders_dict)[i] + ' --> ' + j))
            plt.imshow(d, vmin=mean-n*std, vmax=mean+n*std)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'img_{:s}_{:s}.png'.format(j, list(folders_dict)[i])) )
            # plt.show()
            plt.close('all')

    for l in ['4.0_4.5', '5_5.5', '5.2_5.5', '6_6.5', '7_7.5', '8.5_9', '15_17', '20_22', '25_27']:
        print_stats(statistics, l)

if __name__ == '__main__':
    main()