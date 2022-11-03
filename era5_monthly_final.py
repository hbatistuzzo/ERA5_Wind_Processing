#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:39:42 2021

@author: hbatistuzzo
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
import multiprocessing as mp
import os
import pickle
import time
import humanize
from sys import getsizeof
import time
import matplotlib.font_manager
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
import cmocean
from dask.diagnostics import ProgressBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
print("imported packages...")


def stats(vari):
    mu = np.nanmean(vari)
    sigma = np.nanstd(vari)
    vari_min = np.nanmin(vari)
    vari_max = np.nanmax(vari)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    return mu, sigma, vari_min, vari_max

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


#######################
# from scratch Robinson
import numpy.ma as ma
def shiftgrid(lon0,datain,lonsin,start=True,cyclic=360.0):
    """
    Shift global lat/lon grid east or west.
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    lon0             starting longitude for shifted grid
                     (ending longitude if start=False). lon0 must be on
                     input grid (within the range of lonsin).
    datain           original data with longitude the right-most
                     dimension.
    lonsin           original longitudes.
    ==============   ====================================================
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    start            if True, lon0 represents the starting longitude
                     of the new grid. if False, lon0 is the ending
                     longitude. Default True.
    cyclic           width of periodic domain (default 360)
    ==============   ====================================================
    returns ``dataout,lonsout`` (data and longitudes on shifted grid).
    """
    if np.fabs(lonsin[-1]-lonsin[0]-cyclic) > 1.e-4:
        # Use all data instead of raise ValueError, 'cyclic point not included'
        start_idx = 0
    else:
        # If cyclic, remove the duplicate point
        start_idx = 1
    if lon0 < lonsin[0] or lon0 > lonsin[-1]:
        raise ValueError('lon0 outside of range of lonsin')
    i0 = np.argmin(np.fabs(lonsin-lon0))
    i0_shift = len(lonsin)-i0
    if ma.isMA(datain):
        dataout  = ma.zeros(datain.shape,datain.dtype)
    else:
        dataout  = np.zeros(datain.shape,datain.dtype)
    if ma.isMA(lonsin):
        lonsout = ma.zeros(lonsin.shape,lonsin.dtype)
    else:
        lonsout = np.zeros(lonsin.shape,lonsin.dtype)
    if start:
        lonsout[0:i0_shift] = lonsin[i0:]
    else:
        lonsout[0:i0_shift] = lonsin[i0:]-cyclic
    dataout[...,0:i0_shift] = datain[...,i0:]
    if start:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]+cyclic
    else:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]
    dataout[...,i0_shift:] = datain[...,start_idx:i0+start_idx]
    return dataout,lonsout

ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/global/"
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly_means_stds.nc"
ds = xr.open_dataset(path2)

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

u10n_mean = ds.u10n_mean_months
v10n_mean = ds.v10n_mean_months
u10_mean = ds.u10_mean_months
v10_mean = ds.v10_mean_months


lats = v10_mean[0].lat.values
lons = v10_mean[0].lon.values

# for v10
z = v10_mean.values

v10 = {}
namespace = globals()
v10_list=[]
for m in np.arange(0,12):
    v10[month[m]] = v10_mean[m,:,:].values
    v10_list.append(v10_mean[m,:,:].values) #this works for separating the months
    namespace[f'v10_mean_{month[m]}'] = v10_mean[m] #separates the 12 dataarrays by name
    print('this worked')


#for a fixed colorbar, lets get the mean of the means and the mean of the stds
mu = round(np.nanmean(z),4)
sigma = round(np.nanstd(z),4)
z_min = round(np.nanmin(z),4)
z_max = round(np.nanmax(z),4)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {z_min:.2f}, max is {z_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])
ticks_v10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in v10.keys():
    lons = v10_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = v10[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                     levels=[ssm,sm,m,ms,mss],zorder=1)
    fmt = {} 
    strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
    ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
    ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
    ax.set_global()
    plt.title(f'ERA5 v10 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10,extend='both',pad=0.1)
    cbar.set_label('Meridional wind at 10m (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
    # gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.rotate_labels = False
    gl.ypadding = 30
    gl.xpadding = 10
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-2,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+2,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir0 + f'v10_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'v10_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()

#NICE! Now for movies
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/global/v10/monthly/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/global/v10/monthly/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))


####################################################################################################
# Now we repeat the process for STRESS (iews and inss)
from pylab import text

ldir0= r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/"
ds = xr.open_dataset(ldir0+'stress_monthly_means_stds.nc')


month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

ewss_mean = ds.ewss_mean_months
nsss_mean = ds.nsss_mean_months
iews_mean = ds.iews_mean_months
inss_mean = ds.inss_mean_months


lats = ewss_mean[0].lat.values
lons = ewss_mean[0].lon.values

# for iews
z = iews_mean.values

iews = {}
namespace = globals()
iews_list=[]
for m in np.arange(0,12):
    iews[month[m]] = iews_mean[m,:,:].values
    iews_list.append(iews_mean[m,:,:].values) #this works for separating the months
    namespace[f'iews_mean_{month[m]}'] = iews_mean[m] #separates the 12 dataarrays by name
    print('this worked')


#for a fixed colorbar, lets get the mean of the means and the mean of the stds
mu = round(np.nanmean(z),4)
sigma = round(np.nanstd(z),4)
z_min = round(np.nanmin(z),4)
z_max = round(np.nanmax(z),4)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {z_min:.2f}, max is {z_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])
ticks_iews = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

for mon in iews.keys():
    lons = iews_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = iews[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                     levels=[ssm,sm,m,ms,mss],zorder=1)
    fmt = {} 
    strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
    ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
    ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
    ax.set_global()
    plt.title(f'ERA5 iews 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_iews,extend='both',pad=0.1)
    cbar.set_label('Instantaneous Eastward Turbulent Surface Stress (N m**-2)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_iews,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
    # gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.rotate_labels = False
    gl.ypadding = 30
    gl.xpadding = 10
    cbar.ax.get_yaxis().set_ticks([])
    # cbar.ax.text(ssm-2,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    # cbar.ax.text(mss+2,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    # ax.set_aspect('auto')
    ax.set_aspect('auto')
    text(0, 0, f'MIN = {z_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, 0, f'MAX = {z_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'iews_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'iews_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()

#Stress movies
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/global/iews/new/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/stress/global/iews/new/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))






#### finally the monthly std
from pylab import text

ldir0= r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/"
ds = xr.open_dataset(ldir0+'stress_monthly_means_stds.nc')


month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

ewss_std = ds.ewss_std_months
nsss_std = ds.nsss_std_months
iews_std = ds.iews_std_months
inss_std = ds.inss_std_months


lats = ewss_std[0].lat.values
lons = ewss_std[0].lon.values

# for inss
z = inss_std.values

inss = {}
namespace = globals()
inss_list=[]
for m in np.arange(0,12):
    inss[month[m]] = inss_std[m,:,:].values
    inss_list.append(inss_std[m,:,:].values) #this works for separating the months
    namespace[f'inss_std_{month[m]}'] = inss_std[m] #separates the 12 dataarrays by name
    print('this worked')


#for a fixed colorbar, lets get the mean of the means and the mean of the stds
mu = round(np.nanmean(z),4)
sigma = round(np.nanstd(z),4)
z_min = round(np.nanmin(z),4)
z_max = round(np.nanmax(z),4)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {z_min:.2f}, max is {z_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])
# ticks_inss = [ssm,sm,m,ms,mss]
ticks = [0,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

for mon in inss.keys():
    lons = inss_std[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = inss[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                     levels=[ssm,sm,m,ms,mss],zorder=1)
    fmt = {} 
    strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
    ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
    ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
    ax.set_global()
    plt.title(f'ERA5 inss 1979-2020 {mon} std',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",extend='max',ticks=ticks,pad=0.1)
    cbar.set_label('Instantaneous Northward Turbulent Surface Stress (N m**-2)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks,extend='max')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
    # gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.rotate_labels = False
    gl.ypadding = 30
    gl.xpadding = 10
    cbar.ax.get_yaxis().set_ticks([])
    # cbar.ax.text(ssm-2,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    # cbar.ax.text(mss+2,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    # ax.set_aspect('auto')
    ax.set_aspect('auto')
    text(0, 0, f'MIN = {z_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, 0, f'MAX = {z_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'inss_{mon}_std_global.png')):
        plt.savefig(ldir0 + f'inss_{mon}_std_global.png',bbox_inches='tight')
    plt.show()
plt.show()

#Stress movies
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/global/inss/std/').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/stress/global/inss/std/'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))





#################################################################
# Now for Ilhas (25/01)
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/ilhas/"
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly_means_stds.nc"
ds = xr.open_dataset(path2)

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


#Define the region
latN = 10.25
latS = -10.25
lonW = 295.25
lonE = 15.25

vari = ds.u10_mean_months #12 x 720 x 1440
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

#Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set FIXED colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [sm,m,ms]
ticks_alt = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4]
gfont = {'fontsize' : 20}


var = {}
namespace = globals()
var_list=[]
for n in np.arange(0,12):
    var[month[n]] = vari_mean_ilhas[n,:,:].values
    var_list.append(vari_mean_ilhas[n,:,:].values) #this works for separating the months
    namespace[f'u10_mean_{month[n]}'] = vari_mean_ilhas[n] #separates the 12 dataarrays by name
    print('this worked')

vals = np.array([[vari_min, 0], [0, vari_max]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

figs = []
for mon in var.keys():
    mu, sigma, vari_min, vari_max = stats(var[mon])
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_var = [sm,m,ms]
    ticks_var_cbar = [sm,m,ms]
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
    CS = plt.contour(lon, lat, var[mon], transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
              levels=[sm,m,ms],zorder=1)
    fmt = {} 
    strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
    ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25)
    ax.add_feature(cfeature.RIVERS, linewidths=0.5)
    ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
    scale='50m',facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)

    plt.title(f"ERA5 u10 1993-2019 {mon} mean",fontdict = gfont)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-8,vmax=4,zorder=0,norm=norm)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
    cbar.set_label('Zonal wind at 10m (m/s)',fontsize = 12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='both',pad=0.1,shrink=0.9)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    text(0, -0.75, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, -0.75, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'u10_{mon}_mean_ilhas.png')):
        plt.savefig(ldir0 + f'u10_{mon}_mean_ilhas.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()



##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/ilhas/u10/new').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/ilhas/u10/new'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
    

##needs to customize for v10
vari = ds.v10_mean_months #12 x 720 x 1440
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

#Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set FIXED colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [ssm,sm,m,ms,mss]
ticks_alt = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
gfont = {'fontsize' : 20}


var = {}
namespace = globals()
var_list=[]
for n in np.arange(0,12):
    var[month[n]] = vari_mean_ilhas[n,:,:].values
    var_list.append(vari_mean_ilhas[n,:,:].values) #this works for separating the months
    namespace[f'u10_mean_{month[n]}'] = vari_mean_ilhas[n] #separates the 12 dataarrays by name
    print('this worked')

vals = np.array([[vari_min, 0], [0, vari_max]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

figs = []
for mon in var.keys():
    mu, sigma, vari_min, vari_max = stats(var[mon])
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_var = [ssm,sm,m,ms,mss]
    ticks_var_cbar = [sm,m,ms]
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
    CS = plt.contour(lon, lat, var[mon], transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
              levels=[sm,m,ms],zorder=1)
    fmt = {} 
    strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
    ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25)
    ax.add_feature(cfeature.RIVERS, linewidths=0.5)
    ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
    scale='50m',facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)

    plt.title(f"ERA5 v10 1993-2019 {mon} mean",fontdict = gfont)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-6,vmax=6,zorder=0,norm=norm)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
    cbar.set_label('Meridional wind at 10m (m/s)',fontsize = 12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var_cbar,extend='both',pad=0.1,shrink=0.9)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    text(0, -0.75, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, -0.75, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'v10_{mon}_mean_ilhas.png')):
        plt.savefig(ldir0 + f'v10_{mon}_mean_ilhas.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()


##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/ilhas/v10/new').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/ilhas/v10/new'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
    







########################################
# Finally, the stress

ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/ilhas/"
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/stress_monthly_means_stds.nc"
ds = xr.open_dataset(path2)

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


#Define the region
latN = 10.25
latS = -10.25
lonW = 295.25
lonE = 15.25

vari = ds.iews_mean_months #12 x 720 x 1440
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

#Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set FIXED colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [sm,m,ms]
ticks_alt = np.arange(-0.15,0.15,0.05)
gfont = {'fontsize' : 20}


var = {}
namespace = globals()
var_list=[]
for n in np.arange(0,12):
    var[month[n]] = vari_mean_ilhas[n,:,:].values
    var_list.append(vari_mean_ilhas[n,:,:].values) #this works for separating the months
    namespace[f'u10_mean_{month[n]}'] = vari_mean_ilhas[n] #separates the 12 dataarrays by name
    print('this worked')

vals = np.array([[-0.15, 0], [0, 0.05]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

figs = []
for mon in var.keys():
    mu, sigma, vari_min, vari_max = stats(var[mon])
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=3),
             np.around(mu-sigma,decimals=3),
             np.around(mu,decimals=3),
             np.around(mu+sigma,decimals=3),
             np.around(mu+2*sigma,decimals=3)]
    print([ssm,sm,m,ms,mss])
    ticks_var = [sm,m,ms]
    ticks_var_cbar = [sm,m,ms]
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
    CS = plt.contour(lon, lat, var[mon], transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
              levels=[sm,m,ms],zorder=1)
    fmt = {} 
    strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=7,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
    ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25)
    ax.add_feature(cfeature.RIVERS, linewidths=0.5)
    ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
    scale='50m',facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)

    plt.title(f"ERA5 IEWS 1993-2019 {mon} mean",fontdict = gfont)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-0.15,vmax=0.05,zorder=0,norm=norm)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
    cbar.set_label('Instantaneous Eastward Turbulent Surface Stress (N m**-2)',fontsize = 12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='both',pad=0.1,shrink=0.9)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    text(0, -0.75, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, -0.75, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'iews_{mon}_mean_ilhas.png')):
        plt.savefig(ldir0 + f'iews_{mon}_mean_ilhas.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()



##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/ilhas/iews/monthly').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/stress/ilhas/iews/monthly'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
















###and for inss
vari = ds.inss_mean_months #12 x 720 x 1440
lon = vari.lon.values # 1440 array float32
lat = vari.lat.values # 720 ~~~

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#### the new lon needs to be this size!
lon = vari_mean_ilhas.lon.values 
lat = vari_mean_ilhas.lat.values 

#Get some stats:
mu, sigma, vari_min, vari_max = stats(vari)

# 3) Set FIXED colorbar intervals
[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

# 4) aux
ticks_var = [sm,m,ms]
ticks_alt = np.arange(-0.15,0.15,0.05)
gfont = {'fontsize' : 20}


var = {}
namespace = globals()
var_list=[]
for n in np.arange(0,12):
    var[month[n]] = vari_mean_ilhas[n,:,:].values
    var_list.append(vari_mean_ilhas[n,:,:].values) #this works for separating the months
    namespace[f'u10_mean_{month[n]}'] = vari_mean_ilhas[n] #separates the 12 dataarrays by name
    print('this worked')

vals = np.array([[-0.15, 0], [0, 0.15]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

figs = []
for mon in var.keys():
    mu, sigma, vari_min, vari_max = stats(var[mon])
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=3),
             np.around(mu-sigma,decimals=3),
             np.around(mu,decimals=3),
             np.around(mu+sigma,decimals=3),
             np.around(mu+2*sigma,decimals=3)]
    print([ssm,sm,m,ms,mss])
    ticks_var = [sm,m,ms]
    ticks_var_cbar = [sm,m,ms]
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
    CS = plt.contour(lon, lat, var[mon], transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
              levels=[sm,m,ms],zorder=1)
    fmt = {} 
    strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax.clabel(CS, CS.levels, inline=True,fontsize=7,fmt=fmt)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
    ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_feature(cfeature.BORDERS, linewidths=0.25)
    ax.add_feature(cfeature.RIVERS, linewidths=0.5)
    ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
    scale='50m',facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)

    plt.title(f"ERA5 INSS 1993-2019 {mon} mean",fontdict = gfont)
    cf = plt.pcolormesh(lon,lat,var[mon],transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-0.15,vmax=0.15,zorder=0,norm=norm)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
    cbar.set_label('Instantaneous Northward Turbulent Surface Stress (N m**-2)',fontsize = 12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_var,extend='both',pad=0.1,shrink=0.9)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    text(0, -0.75, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    text(1, -0.75, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
    if not (os.path.exists(ldir0 + f'inss_{mon}_mean_ilhas.png')):
        plt.savefig(ldir0 + f'inss_{mon}_mean_ilhas.png',bbox_inches='tight')
    figs.append([fig])
    plt.show()



##### movie time
import cv2
import argparse
import os
from pathlib import Path
paths = sorted(Path('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/ilhas/inss/monthly').iterdir(), key=os.path.getmtime)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '/media/hbatistuzzo/DATA/Ilhas/Era5/stress/ilhas/inss/monthly'
ext = args['extension']
output = args['output']

images = []
for f in sorted(os.listdir(dir_path),key=os.path.getmtime):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 2.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))

