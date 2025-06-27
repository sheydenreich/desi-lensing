import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u


def get_versions(version):
    if str(version) == "v0.1":
        return {"BGS_BRIGHT":"v0.1",
                "LRG" : "v0.1"}

    elif str(version) == "v0.2":
        return {"BGS_BRIGHT":"v0.5.1",
                "LRG" : "v0.4.5"}

    elif str(version) == "v0.3":
        return {"BGS_BRIGHT":"v0.6",
                "LRG" : "v0.6"}
    
    elif str(version) == "v1.1":
        return {"BGS_BRIGHT":"v1.1",
                "LRG" : "v1.1"}
    
    elif str(version) == "v1.2":
        return {"BGS_BRIGHT":"v1.2",
                "LRG" : "v1.2"}

    else:
        return {"BGS_BRIGHT":version,
                "LRG" : version}

    raise ValueError(f"Version {version} not known, known versions: v0.1, v0.2, v0.3")
  

def clean_read(config,section,option,split,sep=',',convert_to_float=False,
               convert_to_float_range=False,convert_to_int=False,convert_to_bool=False):
    value = config.get(section,option)
    if(split):
        value = value.split(sep)
        value = [remove_spaces(v) for v in value]
        if(convert_to_float):
            value = np.array([float(v) for v in value])
        if(convert_to_float_range):
            value_arr = np.zeros((len(value),2))
            for i,v in enumerate(value):
                value_arr[i,:] = [float(vv) for vv in v.split('-')]
            value = value_arr
        if(convert_to_int):
            value = np.array([int(v) for v in value])
        if(convert_to_bool):
            value = np.array([to_bool(v) for v in value])
    else:
        value = remove_spaces(value)
        if(convert_to_float):
            value = float(value)
        if(convert_to_int):
            value = int(value)
        if(convert_to_float_range):
            value = np.array([float(v) for v in value.split('-')])
        if(convert_to_bool):
            value = to_bool(value)
    return value

def remove_spaces(value):
    while value.startswith(' '):
        value = value[1:]
    while value.endswith(' '):
        value = value[:-1]
    return value

def to_bool(value):
    if(value.lower() in ['true','t','1']):
        return True
    elif(value.lower() in ['false','f','0']):
        return False
    else:
        raise ValueError(f"Value {value} is not a boolean")
    
def get_logger(plot_path,script_name,name):
    import logging
# Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(plot_path+'logfile_'+script_name+'.log',mode='w')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


    return logger

def create_sky_coord(ra, dec, deg = True, ra_u = None, dec_u = None): 
    """
    Creates SkyCoord object for match_catalogs function. 
    
    Parameters 
    ----------
    ra : (list) or (str)
        List containing RAs of the objects. 
        Or RA of the object.
    dec : (list) or (str)
        List containing Decs of the objects. 
        Or Dec of the object. 
    deg : (bool)
        Is the coordinate already in degrees?
    ra_u : (astropy.unit) or (None)
        RA unit.
    dec_u : (astropy.unit) or (None)
        Dec unit. 
        
    
    Returns 
    -------
    sky_coord : (astropy.coordinates.SkyCoord)
        SkyCoord object containing positional information, 
        including units and separation (dimensionless, set to 1).
    """
    if deg == True: 
        sky_coord = SkyCoord(ra*u.deg, dec*u.deg)
    else: 
        sky_coord = SkyCoord(ra, dec, unit = (ra_u, dec_u))
    
    return sky_coord


def put_survey_on_grid(ra,dec,ra_proj,dec_proj,pixels,vertices,
                      unit='deg',smoothing=0.4*u.deg):
    c_survey = SkyCoord(ra,dec, unit = unit)
    c_footprint = SkyCoord(ra_proj, dec_proj, unit = unit)
    idx, sep2d, dist3d = c_footprint.match_to_catalog_sky(c_survey)
    inside = sep2d < smoothing
    return pixels[inside],vertices[inside],inside

def get_vertices_from_pixels(pixels,inside,nside):
    vertices = np.zeros((len(pixels),4,2))
    vertices[:,:,0] = hp.pix2ang(nside,pixels,nest=False,lonlat=True)[0]
    vertices[:,:,1] = hp.pix2ang(nside,pixels,nest=False,lonlat=True)[1]
    return vertices[inside]

def get_mask(galaxy):
    import os
    gal_fname = os.path.join("/global/homes/s/sven/code/", "galaxy_overlaps", "masks", galaxy+".fits")
    gal_mask = hp.read_map(gal_fname).astype(bool)
    return gal_mask

def vertex_with_edge(skmcls, vertices, color=None, vmin=None, vmax=None, **kwargs):
    """Plot polygons (e.g. Healpix vertices)

    Args:
        vertices: cell boundaries in RA/Dec, from getCountAtLocations()
        color: string or matplib color, or numeric array to set polygon colors
        vmin: if color is numeric array, use vmin to set color of minimum
        vmax: if color is numeric array, use vmin to set color of minimum
        **kwargs: matplotlib.collections.PolyCollection keywords
    Returns:
        matplotlib.collections.PolyCollection
    """
    vertices_ = np.empty_like(vertices)
    vertices_[:,:,0], vertices_[:,:,1] = skmcls.proj.transform(vertices[:,:,0], vertices[:,:,1])

    # remove vertices which are split at the outer meridians
    # find variance of vertice nodes large compared to dispersion of centers
    centers = np.mean(vertices, axis=1)
    x, y = skmcls.proj.transform(centers[:,0], centers[:,1])
    var = np.sum(np.var(vertices_, axis=1), axis=-1) / (x.var() + y.var())
    sel = var < 0.05
    vertices_ = vertices_[sel]

    from matplotlib.collections import PolyCollection
    zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
    rasterized = kwargs.pop('rasterized', True)
    alpha = kwargs.pop('alpha', 1)
    # if alpha < 1:
    #     lw = kwargs.pop('lw', 0)
    # else:
    #     lw = kwargs.pop('lw', None)
    coll = PolyCollection(vertices_, zorder=zorder, rasterized=rasterized, alpha=alpha, **kwargs)
    if color is not None:
        coll.set_array(color[sel])
        coll.set_clim(vmin=vmin, vmax=vmax)
    # coll.set_edgecolor("face")
    skmcls.ax.add_collection(coll)
    skmcls.ax.set_rasterization_zorder(zorder)
    return coll
    
def get_boundary_mask(vertices, nside, niter = 1):
    boundary_mask = np.zeros(hp.nside2npix(nside),dtype=bool)
    mask = np.zeros(hp.nside2npix(nside),dtype=bool)
    mask[vertices] = 1
    assert niter>0
    neighbors = vertices
    for i in range(niter):
        neighbors = hp.get_all_neighbours(nside,neighbors)
        neighbors = np.unique(neighbors,axis=1)
    boundary_mask[neighbors] = 1

    boundary_mask = (boundary_mask & (~mask))

    return boundary_mask

def get_fsky(input_mask, threshold=0.1):
    """get the fraction of the observable sky

    Parameters
    ---------
    input_mask: np.ndarray
        healpy array indicating the input mask (0: masked, 1: visible)
    threshold: int
        mask cutoff value
    """
    if(np.issubdtype(input_mask.dtype, np.bool_)):
        return float(np.sum(input_mask)) / len(input_mask)
    return len(input_mask[input_mask > threshold]) / len(input_mask)


def estimate_sky_coverage(ras,decs,nside=1024):
    phi,theta = np.radians(ras),np.radians(90.-decs)
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    mask = np.zeros(hp.nside2npix(nside))
    mask[ipix] = 1
    return get_fsky(mask)

def make_cover_map(ras,decs,nside=1024):
    phi,theta = np.radians(ras),np.radians(90.-decs)
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    mask = np.zeros(hp.nside2npix(nside),dtype=bool)
    mask[ipix] = 1
    return mask

def get_area(input_mask):
    return get_fsky(input_mask)*4*np.pi * (180/np.pi)**2

def get_overlap(mask1,mask2):
    return np.logical_and(mask1,mask2)

def get_mean(ra,dec,val,nside,fill_value=np.nan,print_counts = False):
    phi,theta = np.radians(ra),np.radians(90.-dec)
    ipix = hp.ang2pix(nside,theta,phi,nest=False)
    values = np.bincount(ipix,weights=val,minlength=hp.nside2npix(nside))
    counts = np.bincount(ipix,minlength=hp.nside2npix(nside))
    return np.where(counts>0,values/counts,fill_value),counts

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

def add_colorbar_legend(fig, ax, gs, color_list, name_list,start=0,skip=1):
    if len(name_list) < len(color_list):
        local_color_list = color_list[:len(name_list)]
        local_name_list = name_list
    elif len(name_list) > len(color_list):
        local_color_list = color_list
        local_name_list = name_list[:len(color_list)]
    else:
        local_color_list = color_list
        local_name_list = name_list

    cmap = mpl.colors.ListedColormap(local_color_list)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []

    tick_labels = local_name_list

    ticks = np.linspace(0, 1, len(tick_labels) + 1)
    ticks = 0.5 * (ticks[1:] + ticks[:-1])
    for i in range(ax.shape[0])[start::skip]:
        ax[i,-1] = fig.add_subplot(gs[i:i+skip, -1])

        cb = plt.colorbar(sm, cax=ax[i,-1], pad=0.0, ticks=ticks)
        cb.ax.set_yticklabels(tick_labels)
        cb.ax.minorticks_off()
        cb.ax.tick_params(size=0)

def initialize_gridspec_figure(figsize,nrows,ncols,add_cbar=True,**kwargs):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows,ncols+1 if add_cbar else ncols,
                                width_ratios = [20]*ncols+[1] if add_cbar else [20]*ncols,
                                **kwargs)
        ax = np.empty((nrows,ncols+1 if add_cbar else ncols),dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                ax[i,j] = fig.add_subplot(gs[i,j],sharey=ax[i,0],
                                        sharex=ax[0,j])
                if(j>0):
                    plt.setp(ax[i,j].get_yticklabels(), visible=False)
                if(i<nrows-1):
                    plt.setp(ax[i,j].get_xticklabels(), visible=False)
        return fig,ax,gs

def using_mpl_scatter_density(fig, x, y, cbar=True, cbar_label=None):
    import mpl_scatter_density
    from matplotlib.colors import LinearSegmentedColormap
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)


    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    if(cbar):
        fig.colorbar(density, label=cbar_label)
    return ax

# def add_colorbar_legend(fig, ax, color_list, name_list):
#     cmap = mpl.colors.ListedColormap(color_list)
#     sm = plt.cm.ScalarMappable(cmap=cmap)
#     sm._A = []
#     tick_labels = name_list

#     ticks = np.linspace(0, 1, len(tick_labels) + 1)
#     ticks = 0.5 * (ticks[1:] + ticks[:-1])
#     n_cbars = ax.shape[0]
#     n_axes_sharing = ax.shape[1]-1
#     norm = mpl.colors.Normalize(vmin=0, vmax=1)

#     for i in range(n_cbars):
#         for j in range(n_axes_sharing):
#             ax[i,j].get_shared_x_axes().remove(ax[i,-1])
#             ax[i,j].get_shared_y_axes().remove(ax[i,-1])

#     for i in range(n_cbars):
#         cb = plt.colorbar(sm, cax=ax[i,-1], pad=0.0, ticks=ticks, norm=norm)
#         cb.ax.set_yticklabels(tick_labels)
#         cb.ax.minorticks_off()
#         cb.ax.tick_params(size=0)