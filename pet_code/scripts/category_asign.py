import os
import sys
import yaml
import matplotlib.pyplot as plt

from pet_code.src.util import np
from pet_code.src.util import pd


def read_files_tsv(data_file, num_sm):
    with open(data_file, "r") as infile:
        lines = infile.readlines()

        lines.pop(0)
        last_line = lines[-1]
        col_last_line = last_line.strip().split("\t")
        x_last_line = col_last_line[:num_bin]
        x_list = list(map(float, x_last_line))
        x_values = np.linspace(x_list[0], x_list[-1], len(x_list))
        lines.pop(-1)
        dict_map = {}
        if num_sm == 1:
            histo = np.empty((num_bin, num_total_slab_ref), dtype=np.int32)
            for line in lines:
                cols = line.strip().split("\t")

                mm = int(cols[-3])
                slab = int(cols[-2])
                slab_ref = mm * num_slab + slab
                bins_data = np.array(list(map(float, cols[:-4])))

                histo[:, slab_ref] = bins_data

        else:
            histo = np.empty((num_bin, num_sm, num_mm, num_slab), dtype=np.int32)
            for line in lines:
                cols = line.strip().split("\t")

                sm = int(cols[-4])
                mm = int(cols[-3])
                slab = int(cols[-2])
                asic = int(cols[-1])
                bins_data = np.array(list(map(float, cols[:-4])))

                histo[:, sm, mm, slab] = bins_data
                dict_map[(sm, mm, slab)] = asic

    return histo, dict_map, x_values


def data_max_f(histo):
    return np.max(histo)


def file_norm(histo, num_sm):
    if num_sm == 1:
        data_norm = np.empty((num_bin, num_total_slab_ref), dtype=np.float32)
        for slab_ref in range(num_total_slab_ref):
            if data_max_f(histo[:, slab_ref]) != 0:
                data_norm[:, slab_ref] = histo[:, slab_ref] / data_max_f(
                    histo[:, slab_ref]
                )
            else:
                data_norm[:, slab_ref] = histo[:, slab_ref]

    else:
        cont = 0
        cont_0 = 0
        data_norm = np.empty((num_bin, num_sm, num_mm, num_slab), dtype=np.float32)
        for sm in range(num_sm):
            for mm in range(num_mm):
                for slab in range(num_slab):
                    cont += 1

                    if data_max_f(histo[:, sm, mm, slab]) != 0:
                        data_norm[:, sm, mm, slab] = histo[
                            :, sm, mm, slab
                        ] / data_max_f(histo[:, sm, mm, slab])
                    else:
                        cont_0 += 1
                        data_norm[:, sm, mm, slab] = histo[:, sm, mm, slab]
                    print("---------Histo---------")
                    print(histo[:, sm, mm, slab])
                    print("---------Max---------")
                    print(data_max_f(histo[:, sm, mm, slab]))
                    print("---------Norm---------")
                    print(data_norm[:, sm, mm, slab])

        print(cont)
        print(cont_0)
    return data_norm


def category_asign(histo_ref, histo_inf, num_sm, dict_map, x_values, flag_plot=False):
    dict_final = {}
    best_slab = np.zeros((num_total_slab_ref), np.int32)

    for sm in range(num_sm):
        for mm in range(num_mm):
            for slab in range(num_slab):
                for slab_ref in range(num_total_slab_ref):
                    dif = np.sum(
                        np.abs(histo_inf[:, sm, mm, slab] - histo_ref[:, slab_ref])
                    )
                    best_slab[slab_ref] = dif
                min_slab = np.argmin(best_slab)

                dict_final[dict_map[(sm, mm, slab)]] = min_slab
                if flag_plot:
                    histo_ref_plot = histo_ref[:, min_slab]
                    histo_inf_plot = histo_inf[:, sm, mm, slab]
                    label_ref = min_slab
                    label_inf = dict_map[(sm, mm, slab)]
                    print(histo_ref_plot)
                    print(histo_inf_plot)
                    print("------------------")
                    plt.plot(x_values, histo_ref_plot, label=f"Ref: {label_ref}")
                    plt.plot(
                        x_values,
                        histo_inf_plot,
                        label=f"Inf: {label_inf}, {sm},{mm},{slab}",
                    )
                    plt.legend()

                    plt.show()

    return dict_final


def calculate_system_ch(yam_dict):
    all_chs = []
    for sm in yam_dict["sm_feb_map"]:
        portID, slaveID, febport = yam_dict["sm_feb_map"][sm]
        for t_ch in yam_dict["time_channels"]:
            ch_id = get_absolute_id(portID, slaveID, febport, t_ch)
            all_chs.append(ch_id)

    return all_chs


def get_absolute_id(portID, slaveID, febport, channelID):
    """
    Calculates absolute channel id from
    electronics numbers.
    """
    return 131072 * portID + 4096 * slaveID + 256 * febport + channelID


def add_category_asign():
    cat_dict = category_asign(
        histo_ref_norm, histo_inf_norm, num_sm_inf, dict_map, x_values
    )
    slab_active_ch = set(cat_dict.keys())
    slab_inactive_ch = set(all_chs) - slab_active_ch

    for ch_inactive in slab_inactive_ch:
        cat_dict[ch_inactive] = 0

    return cat_dict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: python program.py file_ref.tsv file_inf.tsv")
        sys.exit(1)
    yaml_file = "/home/LabPC_10/WorkingDir/sw/python_code/petsys_analysis_i3M/pet_code/test_data/SM_mapping_5rings.yaml"
    with open(yaml_file) as old_map_buffer:
        yaml_dict = yaml.safe_load(old_map_buffer)

    all_chs = calculate_system_ch(yaml_dict)

    file_ref_path = sys.argv[1]
    file_inf_path = sys.argv[2]

    data_file_ref = os.path.join(file_ref_path)
    data_file_inf = os.path.join(file_inf_path)
    file_name_only_ref = os.path.basename(file_ref_path)
    file_name_only_inf = os.path.basename(file_inf_path)
    file_name_only_ref, _ = os.path.splitext(file_name_only_ref)
    file_name_only_inf, _ = os.path.splitext(file_name_only_inf)

    read_file_ref = pd.read_csv(data_file_ref, sep="\t")
    read_file_inf = pd.read_csv(data_file_inf, sep="\t")

    file_path_less, ext = os.path.splitext(file_ref_path)
    num_columns = read_file_ref.shape[1]

    num_sm_ref = 1
    num_sm_inf = 120
    num_mm = 16
    num_slab = 8
    num_bin = num_columns - 4
    num_total_slab_ref = num_mm * num_slab

    histo_ref = read_files_tsv(data_file_ref, num_sm_ref)[0]
    histo_inf = read_files_tsv(data_file_inf, num_sm_inf)[0]
    dict_map = read_files_tsv(data_file_inf, num_sm_inf)[1]
    x_values = read_files_tsv(data_file_inf, num_sm_inf)[2]

    histo_ref_norm = file_norm(histo_ref, num_sm_ref)
    histo_inf_norm = file_norm(histo_inf, num_sm_inf)

    cat_dict = add_category_asign()

    print(cat_dict)
    print(len(cat_dict.keys()))

    # Extrae las claves y los valores en arrays de numpy
    keys = np.array(list(cat_dict.keys()), dtype=np.int32)
    values = np.array(list(cat_dict.values()), dtype=np.int32)

    # Combina las claves y los valores en un solo array de numpy
    combined = np.vstack((keys, values)).T

    # Guarda el array combinado en un archivo binario
    combined.tofile(f"cat_dict/cat_dict_{file_name_only_inf}.bin")
