#!/usr/bin/env python3

"""Make position plots under given (conf) calibration conditions and return plots

Usage: pos_monitor.py (--conf CONFFILE) [-j] INPUT ...

Arguments:
    INPUT  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.

Options:
    -j  Join data from multiple files instead of treating them as separate.
"""

import os
import configparser
import re
import fnmatch

from docopt import docopt
from typing import Callable

import matplotlib.pyplot as plt

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.fits    import fit_gaussian
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
# from pet_code.src.slab_nn import neural_net_pcalc
from pet_code.src.util    import np
from pet_code.src.util    import pd
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import energy_weighted_average
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import select_max_energy
from pet_code.src.util    import select_module
from pet_code.src.util    import shift_to_centres
from pet_code.src.util    import slab_indx


from pet_code.scripts.cal_monitor  import cal_and_sel
#from pet_code.scripts.nnflood_maps import bunch_predictions


def is_eng(x):
    return x[1] is ChannelType.ENERGY


def cog_doi(sm_info: list[list]) -> float:
    esum = sum(x[3] for x in filter(is_eng, sm_info))
    return esum / max(filter(is_eng, sm_info), key=lambda x: x[3])[3]

# def red_wrap(batch_size: int, y_file: str, doi_file: str, mm_indx: Callable, local_pos: Callable) -> Callable:
#     b_func, pr_func = bunch_predictions(batch_size, y_file, doi_file, mm_indx, local_pos)
#     def _do_stuff(typ: str, sm_info: list | None=None, ):
#         if 'save' in typ:
#             b_func()
#     nn_predict = neural_net_pcalc("IMAS-1ring", y_file, doi_file, local_pos)
#     nn_pos = neural_net_pcalc(y_file, doi_file, mm_indx, local_pos)
#     def _rec(sm_info: list[list]) -> tuple[float, float]:
#         _, y, doi = nn_pos(sm_info)
#         return y, doi
#     return _rec

def cog_wrap(chan_map: ChannelMap, local_indx: int, power: int) -> Callable:
    yrec     = energy_weighted_average(chan_map.get_plot_position, local_indx, power)
    drec     = cog_doi
    max_slab = select_max_energy(ChannelType.TIME)    
    def _rec(sm_info: list[list]) -> tuple[float, float]:
        y = yrec(sm_info)
        d = drec(sm_info)
        if (mx_tchn := max_slab(sm_info)):
            y -= chan_map.mapping.at[mx_tchn[0], 'local_y']
            return y, d
        raise IndexError
        
    return _rec


def energy_sel_cut(reader, infiles, flag_ref=False):
    max_slab = select_max_energy(ChannelType.TIME)
    energy_mm_dict = {}
    ldat_mm_dict = {"_14.ldat": [12, 13, 14, 15], "_15.ldat": [12, 13, 14, 15], "_16.ldat": [12, 13, 14, 15],
                    "_44.ldat": [8, 9, 10, 11], "_45.ldat": [8, 9, 10, 11], "_46.ldat": [8, 9, 10, 11],
                    "_67.ldat": [4, 5, 6, 7], "_70.ldat": [4, 5, 6, 7], "_73.ldat": [4, 5, 6, 7],
                    "_94.ldat": [0, 1, 2, 3], "_95.ldat": [0, 1, 2, 3], "_96.ldat": [0, 1, 2, 3]}
    if flag_ref:
        for key, mms in ldat_mm_dict.items():
            matching_files = fnmatch.filter(infiles, f"*{key}*")
            print(f'key: {key}')
            print(matching_files)
            if matching_files:
                file = matching_files[0]
                print(file)
                for evt in reader(file):
                    for sm_info in evt:
                        _, eng = get_supermodule_eng(sm_info)
                        slab_id = max_slab(sm_info)[0]
                        sm, mm  = chan_map.get_modules(slab_id)
                        if mm in mms:
                            try:
                                energy_mm_dict[(sm,mm)].append(eng)
                            except KeyError:
                                energy_mm_dict[(sm,mm)] = [eng]
                        else:
                            continue
    else:
        for fn in infiles:
            for evt in reader(file):
                for sm_info in evt:
                    _, eng = get_supermodule_eng(sm_info)
                    slab_id = max_slab(sm_info)[0]
                    sm, mm  = chan_map.get_modules(slab_id)
                    if mm in mms:
                        try:
                            energy_mm_dict[(sm,mm)].append(eng)
                        except KeyError:
                            energy_mm_dict[(sm,mm)] = [eng]
                    else:
                        continue
                        
    for key in energy_mm_dict.keys():
 
        data, bins, otros = plt.hist(energy_mm_dict[key], range=[0, 100], bins=100, label=f"(sm, mm): {key})")
        fit = fit_gaussian(data, bins)

        pars = fit[2]
        amp, mu, sig = pars
        print(f'key: {key}, mu: {mu}, sig: {sig}, amp: {amp}')
        lim_inf = 0.8*mu
        lim_sup = 1.2*mu
        energy_mm_dict[key] = (lim_inf, lim_sup)
        # plt.plot(fit[0], fit[1])
        # plt.savefig(f'{key}')
        # plt.legend()
        # plt.close()
            

    return energy_mm_dict

def position_histograms(batch_size: int      ,
                        ybins    : np.ndarray,
                        dbins    : np.ndarray,
                        pos_rec  : Callable  ,
                        ch_per_mm: int       ,
                        emin_max : dict     ,
                        chan_map : ChannelMap
                        ) -> Callable:
    max_slab = select_max_energy(ChannelType.TIME)
    # erange   = select_energy_range(*emin_max)
    # Channel ordering
    icols    = ['supermodule', 'minimodule', 'local_y']
    isTime   = chan_map.mapping.type.map(lambda x: x is ChannelType.TIME)
    ord_chan = chan_map.mapping[isTime].sort_values(icols).groupby(icols[:-1]).head(ch_per_mm).index
    chan_idx = {id: idx % ch_per_mm for idx, id in enumerate(ord_chan)}
    #sm_nums  = chan_map.mapping.supermodule.unique()
    sm_nums = 120
    mm_nums  = chan_map.mapping.minimodule .unique()
    slab_idx_nums = 8
    def _plotter(evt: tuple[list, list]) -> None:
        #print("STARTING EVENT")
        #print(evt)
        for sm_info in evt:
            _plotter.all_count += 1
            _, eng = get_supermodule_eng(sm_info)
            #print(f'sm_info: {sm_info}')
            #print(f'eng: {eng}')
            try:
                slab_id = max_slab(sm_info)[0]
            except TypeError:
                continue
            
            sm, mm  = chan_map.get_modules(slab_id)
            erange   = select_energy_range(*emin_max[(sm,mm)])
            if erange(eng):
                _plotter.ecount += 1   
                #print("ERROR: ", sm_info)  
                #print(max_slab(sm_info))           
                #print(slab_id, chan_map.get_channel_position(slab_id), slab_indx(chan_map.get_channel_position(slab_id)[0]))
                
                slab_idx = slab_indx(chan_map.get_channel_position(slab_id)[0])
                #sm, mm  = chan_map.get_modules(slab_id)
                #sm_mm = (sm, mm)
                sm_mm_slab_idx = (sm, mm, slab_idx)
                #print(sm,mm)
                #print("-------------")
                try:
                    Y, DOI = pos_rec(sm_info)
                except IndexError:
                    continue
                _plotter.sl_count += 1
                #print(Y, DOI, sm_mm)
                _plotter.slab_id[tuple(sm_mm_slab_idx)] = slab_id
                if Y < ybins[-1] and (bn_idx := np.searchsorted(ybins, Y, side='right') - 1) >= 0:
                    _plotter.yspecs[tuple(sm_mm_slab_idx)][bn_idx] += 1
                    if DOI < dbins[-1] and (bn_idx := np.searchsorted(dbins, DOI, side='right') - 1) >= 0:
                        _plotter.dspecs[tuple(sm_mm_slab_idx)][bn_idx] += 1
                _plotter.sl_count = 0
                _plotter.sm_mm_slab_idx.fill(0)
            #print("--------------")
    _plotter.yspecs = {(sm, mm, slab_idx): np.zeros(ybins.shape[0] - 1, np.uint) for slab_idx in range(slab_idx_nums) for mm in mm_nums for sm in range(sm_nums)}
    _plotter.dspecs = {(sm, mm, slab_idx): np.zeros(dbins.shape[0] - 1, np.uint) for slab_idx in range(slab_idx_nums) for mm in mm_nums for sm in range(sm_nums)}
    #_plotter.sm_mm  = np.zeros((batch_size, 2), np.uint)
    _plotter.sm_mm_slab_idx  = np.zeros((batch_size, 3), np.uint)
    _plotter.all_count = 0
    _plotter.ecount    = 0
    _plotter.sl_count  = 0
    _plotter.slab_id   = {(SM, MM, SLAB): 0 for SLAB in range(slab_idx_nums) for MM in mm_nums for SM in range(sm_nums)}
    return _plotter


if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    all_files = args['-j']
    flag_ref = args['-r']
    name_file = args['INPUT']
    
    num_ext = re.search(r"_([0-9]+)\.ldat", name_file[0])
    base_name = os.path.basename(name_file[0])
    base_name = base_name.replace(".ldat", "")

    if num_ext and all_files:
        num_file = 'all'
        print(f"The extracted number is: {num_file}")
    elif num_ext:
        num_file = int(num_ext.group(1))
        print(f"The extracted number is: {num_file}")
    else:
        num_file = base_name
        print(f'No number found in file name. Taking base name {base_name}')

    infiles  = args['INPUT']

    map_file = conf.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)

    min_chan   = conf.getint('filter', 'min_channels')
    #singles    = 'coinc' not in infiles[0]
    singles = False
    evt_filter = filter_event_by_impacts(min_chan, singles=singles)

    time_cal = conf.get     ('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get     ('calibration', 'energy_channels', fallback='')
    eref     = conf.getfloat('calibration', 'energy_reference', fallback=None)
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal, eref=eref)

    emin_max = tuple(map(float, conf.get('output', 'elimits', fallback='42,78').split(',')))
    ybins    = np.arange(*map(float, conf.get('output',   'ybinning', fallback='-13,13,0.2').split(',')))
    dbins    = np.arange(*map(float, conf.get('output', 'doibinning', fallback='0,21,0.2').split(',')))
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reader  = read_petsys_filebyfile(chan_map.ch_type, sm_filter=evt_filter, singles=singles)
    cal_sel = cal_and_sel(cal_func, select_module(chan_map.get_minimodule))

    rec_type = conf.get('output', 'reco_type', fallback='cog')
    batch_size = 1000
    """
    if 'NN' in rec_type:
        nn_yfile = conf.get('network',   'y_file')
        nn_dfile = conf.get('network', 'doi_file')
        batch_size = conf.getint('network', 'batch_size', fallback=  1000)
        system_nm  = conf.get   ('network', 'system_name', fallback='IMAS-1ring')
        # pos_rec  = red_wrap(nn_yfile, nn_dfile, chan_map.get_minimodule_index, chan_map.get_plot_position)
        pos_rec  = bunch_predictions(system_nm, batch_size, nn_yfile, nn_dfile, chan_map.get_minimodule_index, chan_map.get_plot_position, loc_trans=False)
    else:
    """
    # yrec = energy_weighted_average(chan_map.get_plot_position, 1, 2)
    # drec = cog_doi
    pos_rec = cog_wrap(chan_map, 1, 2)
    # plotter = position_histograms(ybins, dbins, yrec, drec, 8, emin_max, chan_map)

    
    energy_cut_dict = energy_sel_cut(reader, infiles, flag_ref)
    print(energy_cut_dict)

    plotter = position_histograms(batch_size, ybins, dbins, pos_rec, 8, energy_cut_dict, chan_map)
    for fn in infiles:
        print(f'Reading {fn}')
        for evt in map(cal_sel, reader(fn)):
            plotter(evt)
    if plotter.sl_count != 0:
        print('doing a last prediction')
        xy_pos, DOI = pos_rec[1]()
        for i in range(plotter.sl_count):
            if xy_pos[i][1] < ybins[-1] and (bn_idx := np.searchsorted(ybins, xy_pos[i][1], side='right') - 1) >= 0:
                plotter.yspecs[tuple(plotter.sm_mm_slab_idx[i])][bn_idx] += 1
                if DOI[i] < dbins[-1] and (bn_idx := np.searchsorted(dbins, DOI[i], side='right') - 1) >= 0:
                    plotter.dspecs[tuple(plotter.sm_mm_slab_idx[i])][bn_idx] += 1

    ydf = pd.DataFrame(shift_to_centres(ybins).round(2), columns=['bin_centres_slab_asic'])
    for (sm, mm, slab_idx), hist in plotter.yspecs.items():
        ydf[f'{sm}_{mm}_{slab_idx}_{plotter.slab_id[(sm, mm, slab_idx)]}'] = hist#.sum(axis=1)
        #print(hist)
    out_file = os.path.join(out_dir, f'yspecs_{rec_type}_{num_file}.tsv')
    ydf = ydf.T.reset_index()
    ydf[['SM', 'MM', 'SLAB', 'ASIC']] = ydf['index'].str.split('_').tolist()
    ydf.drop('index', axis=1).sort_values(['SM', 'MM', 'SLAB', 'ASIC']).to_csv(out_file, sep='\t', index=False)

    
    # ydf['ASCI'] = ydf.apply(lambda row: plotter.slab_id.get((row['SM'], row['MM'], row['SLAB']), None), axis=1)
    # print(plotter.slab_id[(42, 0, 0)])
    # print("Valores en el DataFrame:")
    # print(ydf[['SM', 'MM', 'SLAB']])
    
    # print("DataFrame completo:")
    # print(ydf)


    ddf = pd.DataFrame(shift_to_centres(dbins).round(2), columns=['bin_centres_slab_asic'])
    for (sm, mm, slab_idx), hist in plotter.dspecs.items():
        ddf[f'{sm}_{mm}_{slab_idx}_{plotter.slab_id[(sm, mm, slab_idx)]}'] = hist#.sum(axis=1)
    out_file = os.path.join(out_dir, f'DOIspecs_{rec_type}_{num_file}.tsv')
    ddf = ddf.T.reset_index()
    ddf[['SM', 'MM', 'SLAB','ASIC']] = ddf['index'].str.split('_').tolist()
    ddf.drop('index', axis=1).sort_values(['SM', 'MM', 'SLAB','ASIC']).to_csv(out_file, sep='\t', index=False)

    print(f'All {plotter.all_count}, eng {plotter.ecount}')
    # plt.plot(shift_to_centres(np.arange(0, 300, 10)), plotter.all_spec)
    # plt.show()
