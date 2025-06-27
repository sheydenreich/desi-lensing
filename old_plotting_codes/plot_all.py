from multiprocessing import Pool
import configparser
import sys

from plot_footprints import plot_footprint
from plot_split_footprints import plot_split_footprints
from plot_sys_weights import plot_sys_weights
from plot_ntile_vs_mass import plot_ntile_vs_mass
from plot_splits import plot_split
from plot_datavector import plot_datavector_notomo,plot_bmodes_notomo
from plot_notomo_measurement_vs_compressed import plot_notomo_measurement_vs_compressed
from source_redshift_slope import source_redshift_slope_tomo,source_redshift_slope_notomo
from secondary_effects import all_secondary_effects_tomo,secondary_effects_tomo

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    with Pool(processes=5) as pool:
        pool.apply_async(plot_footprint,[config])
        pool.apply_async(plot_split_footprints,[config])
        pool.apply_async(plot_sys_weights,[config])
        pool.apply_async(plot_ntile_vs_mass,[config])
        pool.apply_async(plot_split,[config])
        pool.apply_async(plot_datavector_notomo,[config])
        pool.apply_async(plot_bmodes_notomo,[config,True])
        pool.apply_async(plot_notomo_measurement_vs_compressed,[config])
        pool.apply_async(source_redshift_slope_tomo,[config,True])
        pool.apply_async(source_redshift_slope_notomo,[config,True])
        pool.apply_async(all_secondary_effects_tomo,[config,True])
        pool.apply_async(secondary_effects_tomo,[config,True])
        pool.close()
        pool.join()