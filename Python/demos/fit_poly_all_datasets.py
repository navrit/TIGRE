from __future__ import division
# import multiprocessing
# from multiprocessing import freeze_support
import shared_functions as s
import numpy as np
import os
from tqdm import tqdm
# from joblib import Parallel, delayed


def do_it(prefix, f, gReconParams, chip_indices):
    base_folder = os.path.join(prefix, f)
    
    base_json_file = os.path.join(base_folder, 'scan_settings.json')
    if not os.path.exists(base_json_file):
        print(f'JSON file does not exist! --> {base_json_file}')
        return
    
    results_folder = os.path.join(base_folder, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    spectral_projs_th0, spectral_open_th0, spectral_projs_th1, spectral_open_th1, _, _, exp_time, _, _, _, _, th0_dacs_list, _ = \
        s.load_or_generate_data_arrays(base_json_file, base_folder, results_folder, gReconParams)

    open_mean_th0_all_dacs = np.mean(spectral_open_th0, axis=1)
    open_mean_th1_all_dacs = np.mean(spectral_open_th1, axis=1)

    for i in range(open_mean_th0_all_dacs.shape[0]):
        # print(i, open_mean_th0_all_dacs.shape, open_mean_th0_all_dacs.shape[0])  # E.g. 0 (9, 512, 512) 9
        open_mean_th0_all_dacs[i, :, :] /= exp_time[i]
        open_mean_th1_all_dacs[i, :, :] /= exp_time[i]

    for i in range(spectral_projs_th0.shape[0]):
        spectral_projs_th0[i, :, :, :] = spectral_projs_th0[i, :, :, :] / exp_time[i]
        spectral_projs_th1[i, :, :, :] = spectral_projs_th1[i, :, :, :] / exp_time[i]

    corrected_DAC_values_from_open_images = s.generate_correct_dac_values(gReconParams, open_mean_th0_all_dacs, th0_dacs_list, chip_indices, plot=False, poly_order=2, open_img_path=results_folder)
    print(corrected_DAC_values_from_open_images.shape)

    output_folder = os.path.join(base_folder, 'fit_poly_all_datasets')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(os.path.join(output_folder, 'all_dac_values.npy'), corrected_DAC_values_from_open_images)


if __name__ == '__main__':
    # freeze_support()
    prefix = os.path.join('f:\\', 'jasper', 'data')
    folders_with_valid_datasets = [
        '20220909_Legoman_nomotion',
        '20220907_Legoman_oldplate',
        '20220905_Legoman_oldplate',
        '20220822_ffpe_WhateverBreast',
        '20220819_ffpe_WhateverBreast',
        '20220818_ffpe_WhateverBreast',
        '20220812_BreastTissueFFPE',
        '20220810_HamNCheese',
    ]

    chip_indices_list = [
        (0,1,2,3),
        (0,1,2,3),
        (2,3,0,1),
        (2,3,0,1),
        (2,3,0,1),
        (2,3,0,1),
        (2,3,0,1),
        (2,3,0,1)
    ]


    gReconParams = dict()
    gReconParams['pixels'] = 512
    gReconParams['fill_gap'] = True
    gReconParams['median_filter'] = False
    gReconParams['bad_pixel_correction'] = True

    for idx, f in tqdm( enumerate(folders_with_valid_datasets)):
        do_it(prefix, f, gReconParams, chip_indices_list[idx])

    # Parallel(n_jobs=len(folders_with_valid_datasets))(delayed(do_it)(prefix, f, gReconParams) for f in tqdm(folders_with_valid_datasets))
    # with multiprocessing.Pool(60) as pool:
    #     processes = [pool.apply_async(do_it, args=(prefix, f, gReconParams)) for f in tqdm(folders_with_valid_datasets)]
    #     result = [p.get() for p in processes]
        
