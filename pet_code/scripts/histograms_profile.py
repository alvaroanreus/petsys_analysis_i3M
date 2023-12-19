import os
import sys
import matplotlib.pyplot as plt

from pet_code.src.util    import np
from pet_code.src.util    import pd

def plot_figures(axs, x_values, data, sm, mm, slab):
        if mm < 4:
            axs[0, mm].plot(x_values, data, label=f'SLAB {slab}')
            axs[0, mm].set_title(f'SM {sm} MM {mm}')
            axs[0, 0].legend()

        elif 4 <= mm < 8:
            axs[1, mm-4].plot(x_values, data, label=f'SLAB {slab}')
            axs[1, mm-4].set_title(f'SM {sm} MM {mm}')

        elif 8 <= mm < 12:
            axs[2, mm-8].plot(x_values, data, label=f'SLAB {slab}')
            axs[2, mm-8].set_title(f'SM {sm} MM {mm}')

        elif 12 <= mm < 16:
            axs[3, mm-12].plot(x_values, data, label=f'SLAB {slab}')
            axs[3, mm-12].set_title(f'SM {sm} MM {mm}')


if len(sys.argv) != 2:
    print("Use: python histograms_profile.py folder_path")
    sys.exit(1)

folder_path = sys.argv[1]

tsv_files = [f for f in os.listdir(folder_path) if f.endswith(".tsv")]


for tsv_file in tsv_files:
    data_file = os.path.join(folder_path, tsv_file)
    read_file = pd.read_csv(data_file, sep='\t')

    tsv_file_less, ext = os.path.splitext(tsv_file)
    num_columns = read_file.shape[1]

    num_sm = 42, 90
    num_sm_idx = 2
    num_mm = 16
    num_slab = 8
    num_bin = num_columns - 4

    save_path = "Histograms_prueba_asic"
    os.makedirs(save_path, exist_ok=True)

    histo = np.empty((num_bin, len(num_sm), num_mm, num_slab), dtype=np.int32)
    histo_norm = np.empty((num_bin, len(num_sm), num_mm, num_slab), dtype=np.int32)

    with open(data_file, "r") as infile:
        lines = infile.readlines()

        header = lines.pop(0)
        last_line = lines[-1]
        col_last_line = last_line.strip().split('\t')
        x_last_line = col_last_line[:num_bin]
        x_list = list(map(float, x_last_line))
        x_values = np.linspace(x_list[0], x_list[-1], len(x_list))
        footer = lines.pop(-1)


        for line in lines:
            cols = line.strip().split('\t')

            sm = int(cols[-4])
            mm = int(cols[-3])
            slab = int(cols[-2])
            
            bins_data = np.array(list(map(float, cols[:-4])))

            sm_index = num_sm.index(sm) 
            histo[:, sm_index, mm, slab] = bins_data
        

    #Normalize x
    x_values_norm = x_values 

    data_max = np.zeros((num_sm_idx, num_mm, num_slab), np.int32)

    for sm in range(num_sm_idx):
        for mm in range(num_mm):
            for slab in range(num_slab):
                data_slabs = histo[:, sm, mm, slab]
                max_value = np.max(data_slabs)
                data_max[sm, mm, slab] = max_value    


    for sm in range(num_sm_idx):
        fig, axs_sm = plt.subplots(4, 4, figsize=(15, 8), sharex=False, sharey=True, num=f'SM {sm}')
        fig_norm, axs_sm_norm = plt.subplots(4, 4, figsize=(15, 8), sharex=False, sharey=True, num=f'SM {sm}_norm')

        for mm in range(num_mm):
            for slab in range(num_slab):
                data = histo[:, sm, mm, slab]

                if data_max[sm, mm, slab] != 0:
                    data_norm = histo[:, sm, mm, slab] / data_max[sm, mm, slab]
                else:
                    data_norm = histo[:, sm, mm, slab]
            
                plot_figures(axs_sm, x_values, data, sm, mm, slab)
                plot_figures(axs_sm_norm, x_values_norm, data_norm, sm, mm, slab)

                    
        filename = os.path.join(save_path, f"Histogram_{tsv_file_less}_SM{sm}.png")
        fig.savefig(filename)
        plt.close(fig)
        filename_norm = os.path.join(save_path, f"Histogram_{tsv_file_less}_norm_SM{sm}.png")
        fig_norm.savefig(filename_norm)
        plt.close(fig_norm)
                    

    plt.tight_layout()



