import configparser
import sys
import os
import skymapper as skm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from plotting_utilities import get_versions,get_boundary_mask,put_survey_on_grid,clean_read,get_logger,make_cover_map,get_area,get_overlap,get_fsky
from data_handler import get_last_mtime
sys.path.append(os.path.abspath('..'))
from load_catalogues import get_lens_table,get_source_table
import astropy.units as u
from datetime import datetime

script_name = 'plot_footprint'

import textwrap

def generate_latex_table(data):
    table_template = textwrap.dedent(r"""
    \begin{{table}}
    \centering
    \begin{{tabular}}{{|l|l|l|}}
    \hline
    \textbf{{\gls{{desiy1}} Tracer}} & \textbf{{Lensing Survey}} & \textbf{{Overlap Area [sq deg]}} \\ \hline
    {content}
    \end{{tabular}}
    \caption{{Overlap areas of \gls{{desiy1}} tracers with various lensing surveys.}}
    \label{{tab:intersection_areas}}
    \end{{table}}
    """)

    content = []
    tracers = sorted(set(key.split('_')[0] for key in data.keys()))
    surveys = sorted(set(key.split('_')[-1] for key in data.keys()))

    for tracer in tracers:
        tracer_rows = []
        for i, survey in enumerate(surveys):
            key = f"{tracer}_{survey}"
            area = data.get(key, "N/A")
            if i == 0:
                tracer_rows.append(fr"\multirow{{4}}{{*}}{{\gls{{{tracer.lower()}}}}} & {survey} & {int(area)} \\ \cline{{2-3}}")
            else:
                tracer_rows.append(fr"                           & {survey} & {int(area)} \\ \cline{{2-3}}")
        content.append("\n".join(tracer_rows))

    table_content = "\n\\hline\n".join(content)
    return table_template.format(content=table_content)

def get_overlaps(config):
    version = clean_read(config,'general','version',split=False)
    versions = get_versions(version)

    survey_list = clean_read(config,'general','lensing_surveys',split=True)
    color_list = clean_read(config,'general','color_list',split=True)
    savepath = clean_read(config,'general','savepath',split=False) + os.sep
    savepath_addon = clean_read(config,script_name,'savepath_addon',split=False)
    vertices_path = clean_read(config,script_name,'vertices_path',split=False)
    os.makedirs(vertices_path,exist_ok=True)

    xpos_list = [clean_read(config,script_name,'pos_'+x,split=True,convert_to_float=True)[0] for x in survey_list]
    ypos_list = [clean_read(config,script_name,'pos_'+x,split=True,convert_to_float=True)[1] for x in survey_list]
    alphas_list = [clean_read(config,script_name,'alpha_'+x,split=False,convert_to_float=True) for x in survey_list]

    smoothing = clean_read(config,script_name,'smoothing',split=False,convert_to_float=True)*u.deg
    sep = clean_read(config,script_name,'sep',split=False,convert_to_float=True)
    nside = clean_read(config,script_name,'nside',split=False,convert_to_int=True)
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(savepath+os.sep+version+os.sep+savepath_addon+os.sep,exist_ok=True)
    logger = get_logger(savepath+os.sep+version+os.sep+savepath_addon+os.sep,script_name,__name__)

    overlaps = {}
    lens_masks = {}
    source_masks = {}
    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        # 4) add data to the map, e.g.
        # make density plot
        nside = 512
        
        lens_table = get_lens_table(galaxy_type,None,None,versions=versions, logger=logger)[0]
        lens_mask = make_cover_map(lens_table["ra"],lens_table["dec"],nside=nside)
        np.save("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/healpix_maps/"+galaxy_type+"_lens_mask.npy",lens_mask)
        print(galaxy_type,get_fsky(lens_mask),get_area(lens_mask))
        lens_masks[galaxy_type] = lens_mask

    for survey in survey_list:
        table = get_source_table(survey,galaxy_type,cut_catalogues_to_DESI=False,logger=logger)[0]
        source_mask = make_cover_map(table["ra"],table["dec"],nside=nside)
        print(survey,get_fsky(source_mask),get_area(source_mask))
        source_masks[survey] = source_mask

    for galaxy_type in ["BGS_BRIGHT","LRG"]:
        for survey in survey_list:
            lens_mask = lens_masks[galaxy_type]
            source_mask = source_masks[survey]
            overlap = get_overlap(lens_mask,source_mask)
            print(galaxy_type,survey,get_fsky(overlap),get_area(overlap))
            overlaps[f"{galaxy_type[:3]}_{survey}"] = get_area(overlap)
    with open(savepath+os.sep+version+os.sep+savepath_addon+os.sep+"overlap_table.tex", "w") as f:
        f.write(generate_latex_table(overlaps))

if __name__ == '__main__':
    config = configparser.ConfigParser()
    if(len(sys.argv)>1):
        config.read(sys.argv[1])
    else:
        config.read("/global/homes/s/sven/code/lensingWithoutBorders/plotting/config_plots.conf")
    get_overlaps(config)