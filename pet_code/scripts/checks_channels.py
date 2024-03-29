"""
Simple script to check the channels which have
appear as successfully calibrated channels in
the output of grp_channel_specs.py.
That is: Data was found for that channel and
the fitter managed to get a peak position for
the 511 keV peak (or other chosen peak).
Run with:
python path/to/script/checks_channels.py <map file> <time channel file> <energy channel file>
where:
    map file: The path and name of the .feather file containing the map for the setup.
    time channel file: The txt file containing the peak positions for time channels.
    energy channel file: The txt file containing the peak positions for energy channels.
"""

import sys

import pandas as pd

from pet_code.src.io import ChannelMap
from pet_code.src.util import ChannelType


if __name__ == "__main__":
    map_file = sys.argv[1]
    slab_file = sys.argv[2]
    eng_file = sys.argv[3]

    time_peaks = pd.read_csv(slab_file, sep="\t").set_index("ID")["MU"]
    eng_peaks = pd.read_csv(eng_file, sep="\t").set_index("ID")["MU"]

    chan_map = ChannelMap(map_file)
    chan_ids = chan_map.mapping.index
    tchans = chan_map.get_chantype_ids(ChannelType.TIME)
    echans = chan_map.get_chantype_ids(ChannelType.ENERGY)
    print(
        f"Checking time channels: {time_peaks.shape[0]} of {tchans.shape[0]} have peak value."
    )
    for (
        id
    ) in (
        tchans
    ):  # filter(lambda x: chan_map.get_channel_type(x) is ChannelType.TIME, chan_ids):
        if id not in time_peaks.index:
            print(
                f"Channel {id} has no peak value. SM{chan_map.get_supermodule(id)}, mM{chan_map.get_minimodule(id)}."
            )
    print(
        f"TPeak min, max, mean: {time_peaks.min()}, {time_peaks.max()}, {time_peaks.mean()}"
    )

    print(
        f"Checking energy channels: {eng_peaks.shape[0]} of {echans.shape[0]} have peak value."
    )
    for (
        id
    ) in (
        echans
    ):  # filter(lambda x: chan_map.get_channel_type(x) is ChannelType.ENERGY, chan_ids):
        if id not in eng_peaks.index:
            print(
                f"Channel {id} has no peak value. SM{chan_map.get_supermodule(id)}, mM{chan_map.get_minimodule(id)}."
            )
    print(
        f"EPeak min, max, mean: {eng_peaks.min()}, {eng_peaks.max()}, {eng_peaks.mean()}"
    )
