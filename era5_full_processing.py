#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:56:28 2020

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
import numpy.ma as ma


def stats(vari):
    mu = np.nanmean(vari)
    sigma = np.nanstd(vari)
    vari_min = np.nanmin(vari)
    vari_max = np.nanmax(vari)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {vari_min:.2f}, max is {vari_max:.2f}')
    return mu, sigma, vari_min, vari_max

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
import warnings
import matplotlib.cbook
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
# from mpl_toolkits.basemap import shiftgrid
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import cmocean
from dask.diagnostics import ProgressBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
print("imported packages...")

#some handy functions for analytics
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/"
path = 'download.nc'
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/full_means_stds.nc"
ds = xr.open_dataset(ldir0+path)

ds_size = humanize.naturalsize(ds.nbytes) #why is it double from the original file size?

u10n = ds.u10n.sel(expver=1)
v10n = ds.v10n.sel(expver=1)
u10 = ds.u10.sel(expver=1)
v10 = ds.v10.sel(expver=1)


tic()
u10n_mean = u10n.mean(dim='time').load()
u10n_std = u10n.std(dim='time').load()
toc()

tic()
v10n_mean = v10n.mean(dim='time').load()
v10n_std = v10n.std(dim='time').load()
toc()

tic()
u10_mean = u10.mean(dim='time').load()
u10_std = u10.std(dim='time').load()
toc()

tic()
v10_mean = v10.mean(dim='time').load()
v10_std = v10.std(dim='time').load()
toc()


lat = u10n_std.latitude.values
lon = u10n_std.longitude.values

ddd = {'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N'}},
       'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E'}}}

from collections import OrderedDict as od
z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['u10n_mean'] = u10n_mean.values
z_c1['v10n_mean'] = v10n_mean.values
z_c1['u10_mean'] = u10_mean.values
z_c1['v10_mean'] = v10_mean.values
z_c1['u10n_std'] = u10n_std.values
z_c1['v10n_std'] = v10n_std.values
z_c1['u10_std'] = u10_std.values
z_c1['v10_std'] = v10_std.values

#I think scale factor has to have 4 decimals
encoding = {}
for key in z_c1.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c1[key],
                          'attrs': {'units': 'm s**-1'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c2.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c2[key],
                          'attrs': {'units': 'm s**-1'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
for key in z_c3.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c3[key],
                          'attrs': {'units': 'm s**-1'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c4.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c4[key],
                          'attrs': {'units': 'm s**-1'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

ds = xr.Dataset.from_dict(ddd)
ds.to_netcdf('full_wind_means_stds.nc', format='NETCDF4',
             encoding=encoding)




#Check:
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/full_ERA_wind_means_stds.nc"
ds = xr.open_dataset(path2)

ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/download.nc"
ds_ori = xr.open_dataset(ldir0)

#Done, now for some plotting


lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values

# u10n MEAN
data_u10n_mean = ds.u10n_mean.values

# fixing the longitude
data_u10n_mean, lon = shiftgrid(180., data_u10n_mean, lon, start=False)

#Plotting:
#NEED TO FIX THE COLORBAR AND PLOTTING SCALE WITH MEAN + STD
fig = plt.figure(figsize=(12, 8), dpi=150)
m = Basemap( projection='mill', resolution='l',
                lon_0=0., lat_0=0. )
x, y = m(*np.meshgrid(lon,lat))
colormesh = m.pcolormesh(x,y,data_u10n_mean,shading='flat',cmap=cmocean.cm.balance)
cbar = m.colorbar(colormesh,location='right')
cbar.set_label('u10n (m/s)',fontdict = {'fontsize' : 20})
ax = cbar.ax
ax.tick_params(labelsize=16)
text = ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Verdana', size=18)
text.set_font_properties(font)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=16)
plt.title('ERA5 u10n 1979-2020 mean',fontdict = {'fontsize' : 24},family='Verdana');
if not (os.path.exists(ldir0+'all_adt_mean.png')):
    plt.savefig('all_adt_mean.png')
plt.show()


# v10n MEAN
data_v10n_mean = ds.v10n_mean.values

    # fixing the longitude
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values
data_v10n_mean, lon = shiftgrid(180., data_v10n_mean, lon, start=False)


    #Plotting:
fig = plt.figure(figsize=(12, 8), dpi=150)
m = Basemap( projection='mill', resolution='l',
                lon_0=0., lat_0=0. )
x, y = m(*np.meshgrid(lon,lat))
colormesh = m.pcolormesh(x,y,data_v10n_mean,shading='flat',cmap=cmocean.cm.balance)
cbar = m.colorbar(colormesh,location='right')
cbar.set_label('v10n (m/s)',fontdict = {'fontsize' : 20})
ax = cbar.ax
ax.tick_params(labelsize=16)
text = ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Verdana', size=18)
text.set_font_properties(font)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=16)
plt.title('ERA5 v10n 1979-2020 mean',fontdict = {'fontsize' : 24},family='Verdana');
if not (os.path.exists(ldir0+'all_adt_mean.png')):
    plt.savefig('all_adt_mean.png')
plt.show()


# u10 MEAN
data_u10_mean = ds.u10_mean.values

    # fixing the longitude
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values
data_u10_mean, lon = shiftgrid(180., data_u10_mean, lon, start=False)


    #Plotting:
fig = plt.figure(figsize=(12, 8), dpi=150)
m = Basemap( projection='mill', resolution='l',
                lon_0=0., lat_0=0. )
x, y = m(*np.meshgrid(lon,lat))
colormesh = m.pcolormesh(x,y,data_u10_mean,shading='flat',cmap=cmocean.cm.balance)
cbar = m.colorbar(colormesh,location='right')
cbar.set_label('u10 (m/s)',fontdict = {'fontsize' : 20})
ax = cbar.ax
ax.tick_params(labelsize=16)
text = ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Verdana', size=18)
text.set_font_properties(font)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=16)
plt.title('ERA5 u10 1979-2020 mean',fontdict = {'fontsize' : 24},family='Verdana');
if not (os.path.exists(ldir0+'all_adt_mean.png')):
    plt.savefig('all_adt_mean.png')
plt.show()



# v10 MEAN
data_v10_mean = ds.v10_mean.values

    # fixing the longitude
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values
data_v10_mean, lon = shiftgrid(180., data_v10_mean, lon, start=False)


    #Plotting:
fig = plt.figure(figsize=(12, 8), dpi=150)
m = Basemap( projection='mill', resolution='l',
                lon_0=0., lat_0=0. )
x, y = m(*np.meshgrid(lon,lat))
colormesh = m.pcolormesh(x,y,data_v10_mean,shading='flat',cmap=cmocean.cm.balance)
cbar = m.colorbar(colormesh,location='right')
cbar.set_label('v10 (m/s)',fontdict = {'fontsize' : 20})
ax = cbar.ax
ax.tick_params(labelsize=16)
text = ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Verdana', size=18)
text.set_font_properties(font)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=16)
plt.title('ERA5 v10 1979-2020 mean',fontdict = {'fontsize' : 24},family='Verdana');
if not (os.path.exists(ldir0+'all_adt_mean.png')):
    plt.savefig('all_adt_mean.png')
plt.show()













###############################################################################
#Now for more advanced plots


# u10n MEAN
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values
u10n = ds.u10n_mean.values

u10n, lon = shiftgrid(180., u10n, lon, start=False)

mu = np.nanmean(ds.u10n_mean.values)
sigma = np.nanstd(ds.u10n_mean.values)
u10n_min = np.nanmin(ds.u10n_mean.values)
u10n_max =np.nanmax(ds.u10n_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_min:.2f}, max is {u10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.25)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10n 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,u10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10n,pad=0.13)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10n)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# cbar2.ax.xaxis.set_label_position('top')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.5, color='black', alpha=0.5, linestyle='solid')
gl.xlabels_top = False
gl.ylabels_right = True
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')

# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()






# v10n MEAN
lon = ds.v10n_mean.lon.values
lat = ds.v10n_mean.lat.values
v10n = ds.v10n_mean.values

v10n, lon = shiftgrid(180., v10n, lon, start=False)

mu = np.nanmean(ds.v10n_mean.values)
sigma = np.nanstd(ds.v10n_mean.values)
v10n_min = np.nanmin(ds.v10n_mean.values)
v10n_max =np.nanmax(ds.v10n_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.25)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10n 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,v10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10n,pad=0.13)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10n)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# cbar2.ax.xaxis.set_label_position('top')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.5, color='black', alpha=0.5, linestyle='solid')
gl.xlabels_top = False
gl.ylabels_right = True
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{v10n_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{v10n_max:.2f}',ha='center',va='center')

# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()


# u10 MEAN
lon = ds.u10_mean.lon.values
lat = ds.u10_mean.lat.values
u10 = ds.u10_mean.values

u10, lon = shiftgrid(180., u10, lon, start=False)

mu = np.nanmean(ds.u10_mean.values)
sigma = np.nanstd(ds.u10_mean.values)
u10_min = np.nanmin(ds.u10_mean.values)
u10_max =np.nanmax(ds.u10_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_min:.2f}, max is {u10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.25)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,u10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10,pad=0.13)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# cbar2.ax.xaxis.set_label_position('top')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.5, color='black', alpha=0.5, linestyle='solid')
gl.xlabels_top = False
gl.ylabels_right = True
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{u10_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{u10_max:.2f}',ha='center',va='center')

# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()



# v10 MEAN
lon = ds.v10_mean.lon.values
lat = ds.v10_mean.lat.values
v10 = ds.v10_mean.values

v10, lon = shiftgrid(180., v10, lon, start=False)

mu = np.nanmean(ds.v10_mean.values)
sigma = np.nanstd(ds.v10_mean.values)
v10_min = np.nanmin(ds.v10_mean.values)
v10_max =np.nanmax(ds.v10_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_min:.2f}, max is {v10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.25)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,v10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
                    vmin=ssm,vmax=mss)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10,pad=0.13)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# cbar2.ax.xaxis.set_label_position('top')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.5, color='black', alpha=0.5, linestyle='solid')
gl.xlabels_top = False
gl.ylabels_right = True
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{v10_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{v10_max:.2f}',ha='center',va='center')

# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()


















###############################################################################W
# Now with Mollweide
#Check:
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/full_wind_means_stds.nc"
ds = xr.open_dataset(path2)

# u10n MEAN
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values
u10n = ds.u10n_mean.values

u10n, lon = shiftgrid(180., u10n, lon, start=False) #0 to 360 needs wrapping

mu = np.nanmean(ds.u10n_mean.values)
sigma = np.nanstd(ds.u10n_mean.values)
u10n_min = np.nanmin(ds.u10n_mean.values)
u10n_max =np.nanmax(ds.u10n_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_min:.2f}, max is {u10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, u10n, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10n 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,u10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10n,pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10n)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral zonal wind at 10m (m/s)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()

# v10n MEAN
lon = ds.v10n_mean.lon.values
lat = ds.v10n_mean.lat.values
v10n = ds.v10n_mean.values

v10n, lon = shiftgrid(180., v10n, lon, start=False)

mu = np.nanmean(ds.v10n_mean.values)
sigma = np.nanstd(ds.v10n_mean.values)
v10n_min = np.nanmin(ds.v10n_mean.values)
v10n_max =np.nanmax(ds.v10n_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, v10n, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10n 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,v10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10n,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10n,extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{v10n_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{v10n_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral meridional wind at 10m (m/s)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()

# u10 MEAN
lon = ds.u10_mean.lon.values
lat = ds.u10_mean.lat.values
u10 = ds.u10_mean.values

u10, lon = shiftgrid(180., u10, lon, start=False)

mu = np.nanmean(ds.u10_mean.values)
sigma = np.nanstd(ds.u10_mean.values)
u10_min = np.nanmin(ds.u10_mean.values)
u10_max =np.nanmax(ds.u10_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_min:.2f}, max is {u10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, u10, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,u10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10,pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{u10_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{u10_max:.2f}',ha='center',va='center')
cbar.set_label('Zonal wind at 10m (m/s)')
# if not (os.path.exists(ldir0 + 'u10_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10_full_mean_SA.png',bbox_inches='tight')
plt.show()

# v10 MEAN
lon = ds.v10_mean.lon.values
lat = ds.v10_mean.lat.values
v10 = ds.v10_mean.values

v10, lon = shiftgrid(180., v10, lon, start=False)

mu = np.nanmean(ds.v10_mean.values)
sigma = np.nanstd(ds.v10_mean.values)
v10_min = np.nanmin(ds.v10_mean.values)
v10_max =np.nanmax(ds.v10_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_min:.2f}, max is {v10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, v10, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,v10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10,extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{v10_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{v10_max:.2f}',ha='center',va='center')
cbar.set_label('Meridional wind at 10m (m/s)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()


###############################################################################
#Now for the Ilhas region
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/full_wind_means_stds.nc"
ds = xr.open_dataset(path2)

vari = ds.v10n_mean #12 x 720 x 1440
# u10n MEAN
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values

#u10n
#lets plot for the Archipelago
latN = 10.25
latS = -10.25
lonW = 295.25
lonE = 15.25

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#extract lat and lons
lats = vari_mean_ilhas.lat.values
lons = vari_mean_ilhas.lon.values

mu, sigma, vari_min, vari_max = stats(vari)

# mu = np.nanmean(u10n) #0.4754307347234847
# sigma = np.nanstd(u10n) #0.12794454349812165
# u10n_min = np.nanmin(u10n)
# u10n_max = np.nanmax(u10n)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]

# print([ssm,sm,m,ms,mss])


#for the colorbar levels
ticks = [sm,m,ms]
ticks_alt = [-3,-2,-1,0,1,2,3,4]
gfont = {'fontsize' : 16}

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


vals = np.array([[-3., 0], [0, 4]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

#Adapted for u10n
fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.5)
ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
CS = plt.contour(lons, lats, vari, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                  levels=[sm,m,ms],zorder=1)
fmt = {} 
strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
plt.title("ERA5 v10n 1979-2020 mean",fontdict = gfont)
cf = plt.pcolormesh(lons,lats,vari,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-3,vmax=4,zorder=0,norm=norm)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
cbar.set_label('Neutral Meridional wind at 10m (m/s)',fontsize = 12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks,extend='both',pad=0.1,shrink=0.9)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
gl.xlabels_top = False
gl.ylabels_right = False
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
cbar.ax.get_yaxis().set_ticks([])
text(0, -0.8, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
text(1, -0.8, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
# if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
#     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
plt.show()


















# #v10n
# #lets plot for the Archipelago
# latN = 10.25
# latS = -10.25
# lonW = 295.25
# lonE = 15.25

# v10n_mean1 = ds.v10n_mean.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
# v10n_mean2 = ds.v10n_mean.sel(lat=slice(latN,latS), lon=slice(0,lonE))
# v10n_mean_ilhas = xr.concat([v10n_mean1, v10n_mean2], dim='lon')
# v10n = v10n_mean_ilhas.values

# #extract lat and lons
# lats = v10n_mean_ilhas.lat.values
# lons = v10n_mean_ilhas.lon.values

# mu = np.nanmean(v10n) #0.4754307347234847
# sigma = np.nanstd(v10n) #0.12794454349812165
# v10n_min = np.nanmin(v10n)
# v10n_max = np.nanmax(v10n)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')

# [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
#               np.around(mu-sigma,decimals=2),
#               np.around(mu,decimals=2),
#               np.around(mu+sigma,decimals=2),
#               np.around(mu+2*sigma,decimals=2)]
# print([ssm,sm,m,ms,mss])


# #for the colorbar levels
# ticks_v10n = [ssm,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 16}

# fig = plt.figure(figsize=(10, 6), dpi=400)
# ax = plt.axes(projection=ccrs.PlateCarree())
# fuck = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
# ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
# ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
# ax.add_feature(cfeature.BORDERS, linewidths=0.25)
# ax.add_feature(cfeature.RIVERS, linewidths=0.5)
# ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
# states_provinces = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='admin_1_states_provinces_lines',
#     scale='50m',
#     facecolor='none')
# ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
# plt.title("ERA5 v10n 1979-2020 mean",fontdict = gfont)
# cf = plt.pcolormesh(lons,lats,v10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
#                       vmin=ssm,vmax=mss)
# cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10n,extend='both',pad=0.15,shrink=0.9)
# cbar.set_label('Neutral meridional wind at 10m (m/s)')
# pos = cbar.ax.get_position()
# cbar.ax.set_aspect('auto')
# ax2 = cbar.ax.twiny()
# cbar.ax.set_position(pos)
# ax2.set_position(pos)
# cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10n,extend='both',pad=0.15,shrink=0.9)
# cbar2.ax.xaxis.set_ticks_position('top')
# cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
#                           '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
# gl.xlabels_top = False
# gl.ylabels_right = False
# # gl.xlines = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
# cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(ssm-abs((0.2*ssm)),0.3,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+abs((0.3*mss)),0.3,f'MAX\n{v10n_max:.2f}',ha='center',va='center')
# # if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
# #     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
# plt.show()



# #u10
# #lets plot for the Archipelago
# latN = 10.25
# latS = -10.25
# lonW = 295.25
# lonE = 15.25

# u10_mean1 = ds.u10_mean.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
# u10_mean2 = ds.u10_mean.sel(lat=slice(latN,latS), lon=slice(0,lonE))
# u10_mean_ilhas = xr.concat([u10_mean1, u10_mean2], dim='lon')
# u10 = u10_mean_ilhas.values

# #extract lat and lons
# lats = u10_mean_ilhas.lat.values
# lons = u10_mean_ilhas.lon.values

# mu = np.nanmean(u10) #0.4754307347234847
# sigma = np.nanstd(u10) #0.12794454349812165
# u10_min = np.nanmin(u10)
# u10_max = np.nanmax(u10)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_min:.2f}, max is {u10_max:.2f}')

# [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
#               np.around(mu-sigma,decimals=2),
#               np.around(mu,decimals=2),
#               np.around(mu+sigma,decimals=2),
#               np.around(mu+2*sigma,decimals=2)]
# print([ssm,sm,m,ms,mss])


# #for the colorbar levels
# ticks_u10 = [ssm,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 16}

# fig = plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# fuck = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
# ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
# ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
# ax.add_feature(cfeature.BORDERS, linewidths=0.25)
# ax.add_feature(cfeature.RIVERS, linewidths=0.5)
# ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
# states_provinces = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='admin_1_states_provinces_lines',
#     scale='50m',
#     facecolor='none')
# ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
# plt.title("ERA5 u10 1979-2020 mean",fontdict = gfont)
# cf = plt.pcolormesh(lons,lats,u10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
#                     vmin=ssm,vmax=mss)
# cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10,extend='both',pad=0.15,shrink=0.9)
# cbar.set_label('Zonal meridional wind at 10m (m/s)')
# pos = cbar.ax.get_position()
# cbar.ax.set_aspect('auto')
# ax2 = cbar.ax.twiny()
# cbar.ax.set_position(pos)
# ax2.set_position(pos)
# cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10,extend='both',pad=0.15,shrink=0.9)
# cbar2.ax.xaxis.set_ticks_position('top')
# cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
#                           '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
# gl.xlabels_top = False
# gl.ylabels_right = False
# # gl.xlines = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
# cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(ssm-abs((0.3*ssm)),0.3,f'MIN\n{u10_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+abs((0.3*ssm)),0.3,f'MAX\n{u10_max:.2f}',ha='center',va='center')
# # if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
# #     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
# plt.show()


# #v10
# #lets plot for the Archipelago
# latN = 10.25
# latS = -10.25
# lonW = 295.25
# lonE = 15.25

# v10_mean1 = ds.v10_mean.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
# v10_mean2 = ds.v10_mean.sel(lat=slice(latN,latS), lon=slice(0,lonE))
# v10_mean_ilhas = xr.concat([v10_mean1, v10_mean2], dim='lon')
# v10 = v10_mean_ilhas.values

# #extract lat and lons
# lats = v10_mean_ilhas.lat.values
# lons = v10_mean_ilhas.lon.values

# mu = np.nanmean(v10) #0.4754307347234847
# sigma = np.nanstd(v10) #0.12794454349812165
# v10_min = np.nanmin(v10)
# v10_max = np.nanmax(v10)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_min:.2f}, max is {v10_max:.2f}')

# [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
#               np.around(mu-sigma,decimals=2),
#               np.around(mu,decimals=2),
#               np.around(mu+sigma,decimals=2),
#               np.around(mu+2*sigma,decimals=2)]
# print([ssm,sm,m,ms,mss])


# #for the colorbar levels
# ticks_v10 = [ssm,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 16}

# fig = plt.figure(figsize=(10, 6), dpi=400)
# ax = plt.axes(projection=ccrs.PlateCarree())
# fuck = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
# ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
# ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
# ax.add_feature(cfeature.BORDERS, linewidths=0.25)
# ax.add_feature(cfeature.RIVERS, linewidths=0.5)
# ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
# states_provinces = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='admin_1_states_provinces_lines',
#     scale='50m',
#     facecolor='none')
# ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
# plt.title("ERA5 v10 1979-2020 mean",fontdict = gfont)
# cf = plt.pcolormesh(lons,lats,v10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.balance,
#                       vmin=ssm,vmax=mss)
# cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10,extend='both',pad=0.15,shrink=0.9)
# cbar.set_label('Meridional wind at 10m (m/s)')
# pos = cbar.ax.get_position()
# cbar.ax.set_aspect('auto')
# ax2 = cbar.ax.twiny()
# cbar.ax.set_position(pos)
# ax2.set_position(pos)
# cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10,extend='both',pad=0.15,shrink=0.9)
# cbar2.ax.xaxis.set_ticks_position('top')
# cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
#                           '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
# gl.xlabels_top = False
# gl.ylabels_right = False
# # gl.xlines = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
# cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(ssm-abs((0.2*ssm)),0.3,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(mss+abs((0.3*mss)),0.3,f'MAX\n{v10_max:.2f}',ha='center',va='center')
# # if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
# #     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
# plt.show()
















#### Now for the wind stress ####

ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/stress.nc"
ds = xr.open_dataset(ldir0)

ewss = ds.ewss.sel(expver=1) #Eastward Turbulent Surface Stress
iews = ds.iews.sel(expver=1) #Instantaneous Eastward Turbulent Surface Stress
inss = ds.inss.sel(expver=1) #Instantaneous Northward Turbulent Surface Stress
nsss = ds.nsss.sel(expver=1) #Northward turbulent surface stress


tic()
ewss_mean = ewss.mean(dim='time').load()
ewss_std = ewss.std(dim='time').load()
toc()

tic()
iews_mean = iews.mean(dim='time').load()
iews_std = iews.std(dim='time').load()
toc()

tic()
inss_mean = inss.mean(dim='time').load()
inss_std = inss.std(dim='time').load()
toc()

tic()
nsss_mean = nsss.mean(dim='time').load()
nsss_std = nsss.std(dim='time').load()
toc()


lat = ewss_mean.latitude.values
lon = ewss_mean.longitude.values

ddd = {'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N'}},
       'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E'}}}

from collections import OrderedDict as od
z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['ewss_mean'] = ewss_mean.values
z_c1['nsss_mean'] = nsss_mean.values
z_c2['iews_mean'] = iews_mean.values
z_c2['inss_mean'] = inss_mean.values
z_c3['ewss_std'] = ewss_std.values
z_c3['nsss_std'] = nsss_std.values
z_c4['iews_std'] = iews_std.values
z_c4['inss_std'] = inss_std.values


#I think scale factor has to have 4 decimals
encoding = {}
for key in z_c1.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c1[key],
                          'attrs': {'units': 'N m**-2 s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c2.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c2[key],
                          'attrs': {'units': 'N m**-2'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c3.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c3[key],
                          'attrs': {'units': 'N m**-2 s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c4.keys():
    ddd.update({key: {'dims': ('lat', 'lon'), 'data': z_c4[key],
                          'attrs': {'units': 'N m**-2'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
ds = xr.Dataset.from_dict(ddd)
ds.to_netcdf('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/full_ERA_stress_means_stds.nc', format='NETCDF4',
             encoding=encoding)

#check:
ldir1 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/full_ERA_stress_means_stds.nc"
ds = xr.open_dataset(ldir1)

# ewss MEAN
lon = ds.ewss_mean.lon.values
lat = ds.ewss_mean.lat.values
ewss = ds.ewss_mean.values

ewss, lon = shiftgrid(180., ewss, lon, start=False)

mu = np.nanmean(ds.ewss_mean.values)
sigma = np.nanstd(ds.ewss_mean.values)
ewss_min = np.nanmin(ds.ewss_mean.values)
ewss_max =np.nanmax(ds.ewss_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {ewss_min:.2f}, max is {ewss_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_ewss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, ewss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 ewss 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,ewss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_ewss,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_ewss,extend='both',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{ewss_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{ewss_max:.2f}',ha='center',va='center')
cbar.set_label('Eastward Turbulent Surface Stress (N m**-2 s)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()


# nsss MEAN
lon = ds.nsss_mean.lon.values
lat = ds.nsss_mean.lat.values
nsss = ds.nsss_mean.values

nsss, lon = shiftgrid(180., nsss, lon, start=False)

mu = np.nanmean(ds.nsss_mean.values)
sigma = np.nanstd(ds.nsss_mean.values)
nsss_min = np.nanmin(ds.nsss_mean.values)
nsss_max =np.nanmax(ds.nsss_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {nsss_min:.2f}, max is {nsss_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_nsss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, nsss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                  linewidths=0.5,colors='k',zorder=1,inline=True)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 nsss 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,nsss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_nsss,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_nsss,extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{nsss_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{nsss_max:.2f}',ha='center',va='center')
cbar.set_label('Northward turbulent surface stress (N m**-2 s)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()



# inss MEAN
lon = ds.inss_mean.lon.values
lat = ds.inss_mean.lat.values
inss = ds.inss_mean.values

inss, lon = shiftgrid(180., inss, lon, start=False)

mu = np.nanmean(ds.inss_mean.values)
sigma = np.nanstd(ds.inss_mean.values)
inss_min = np.nanmin(ds.inss_mean.values)
inss_max =np.nanmax(ds.inss_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {inss_min:.2f}, max is {inss_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_inss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, inss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 inss 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,inss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_inss,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_inss,extend='both',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{inss_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{inss_max:.2f}',ha='center',va='center')
cbar.set_label('Instantaneous northward turbulent surface stress (N m**-2)')
# if not (os.path.exists(ldir0 + 'u10_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10_full_mean_SA.png',bbox_inches='tight')
plt.show()

# iews MEAN
lon = ds.iews_mean.lon.values
lat = ds.iews_mean.lat.values
iews = ds.iews_mean.values

iews, lon = shiftgrid(180., iews, lon, start=False)

mu = np.nanmean(ds.iews_mean.values)
sigma = np.nanstd(ds.iews_mean.values)
iews_min = np.nanmin(ds.iews_mean.values)
iews_max =np.nanmax(ds.iews_mean.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {iews_min:.2f}, max is {iews_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_iews = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

CS = plt.contour(lon, lat, iews, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.5,colors='k',zorder=1,inline=True)
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
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 iews 1979-2020 mean",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,iews,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_iews,extend='both',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_iews,extend='both',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                          '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.ax.text(ssm-abs((0.2*ssm)),0,f'MIN\n{iews_min:.2f}',ha='center',va='center')
cbar.ax.text(mss+abs((0.2*ssm)),0,f'MAX\n{iews_max:.2f}',ha='center',va='center')
cbar.set_label('Instantaneous Eastward Turbulent Surface Stress (N m**-2)')
# if not (os.path.exists(ldir0 + 'u10n_full_mean_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_mean_SA.png',bbox_inches='tight')
plt.show()





#and for the std


# ewss std
lon = ds.ewss_std.lon.values
lat = ds.ewss_std.lat.values
ewss = ds.ewss_std.values

ewss, lon = shiftgrid(180., ewss, lon, start=False)

mu = np.nanmean(ds.ewss_std.values)
sigma = np.nanstd(ds.ewss_std.values)
ewss_min = np.nanmin(ds.ewss_std.values)
ewss_max =np.nanmax(ds.ewss_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {ewss_min:.2f}, max is {ewss_max:.2f}')


[m,ms,mss] = [np.around(mu,decimals=2),np.around(mu+sigma,decimals=2),
                       np.around(mu+2*sigma,decimals=2)]
print([m,ms,mss])

#for the colorbar levels
ticks_ewss = [m,ms,mss,ewss_max]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, ewss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 ewss 1979-2020 std",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,ewss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=ewss_max,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_ewss,extend='max',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_ewss,extend='max',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$', 'max']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.set_label('Eastward Turbulent Surface Stress (N m**-2 s)')
# if not (os.path.exists(ldir0 + 'u10n_full_std_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_std_SA.png',bbox_inches='tight')
plt.show()


# nsss std
lon = ds.nsss_std.lon.values
lat = ds.nsss_std.lat.values
nsss = ds.nsss_std.values

nsss, lon = shiftgrid(180., nsss, lon, start=False)

mu = np.nanmean(ds.nsss_std.values)
sigma = np.nanstd(ds.nsss_std.values)
nsss_min = np.nanmin(ds.nsss_std.values)
nsss_max =np.nanmax(ds.nsss_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {nsss_min:.2f}, max is {nsss_max:.2f}')


[m,ms,mss] = [np.around(mu,decimals=2),np.around(mu+sigma,decimals=2),
                       np.around(mu+2*sigma,decimals=2)]
print([m,ms,mss])

#for the colorbar levels
ticks_nsss = [m,ms,mss,nsss_max]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, nsss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 nsss 1979-2020 std",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,nsss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=nsss_max,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_nsss,extend='max',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_nsss,extend='max',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$', 'max']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.set_label('Northward Turbulent Surface Stress (N m**-2 s)')
# if not (os.path.exists(ldir0 + 'u10n_full_std_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_std_SA.png',bbox_inches='tight')
plt.show()


# inss std
lon = ds.inss_std.lon.values
lat = ds.inss_std.lat.values
inss = ds.inss_std.values

inss, lon = shiftgrid(180., inss, lon, start=False)

mu = np.nanmean(ds.inss_std.values)
sigma = np.nanstd(ds.inss_std.values)
inss_min = np.nanmin(ds.inss_std.values)
inss_max =np.nanmax(ds.inss_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {inss_min:.2f}, max is {inss_max:.2f}')


[m,ms,mss] = [np.around(mu,decimals=2),np.around(mu+sigma,decimals=2),
                       np.around(mu+2*sigma,decimals=2)]
print([m,ms,mss])

#for the colorbar levels
ticks_inss = [0,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, inss, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 inss 1979-2020 std",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,inss,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_inss,extend='max',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_inss,extend='max',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.set_label('Instantaneous northward turbulent surface stress (N m**-2)')
# if not (os.path.exists(ldir0 + 'u10n_full_std_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_std_SA.png',bbox_inches='tight')
plt.show()

# iews std
lon = ds.iews_std.lon.values
lat = ds.iews_std.lat.values
iews = ds.iews_std.values

iews, lon = shiftgrid(180., iews, lon, start=False)

mu = np.nanmean(ds.iews_std.values)
sigma = np.nanstd(ds.iews_std.values)
iews_min = np.nanmin(ds.iews_std.values)
iews_max =np.nanmax(ds.iews_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {iews_min:.2f}, max is {iews_max:.2f}')


[m,ms,mss] = [np.around(mu,decimals=2),np.around(mu+sigma,decimals=2),
                       np.around(mu+2*sigma,decimals=2)]
print([m,ms,mss])

#for the colorbar levels
ticks_iews = [0,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=400)
ax = plt.axes(projection=ccrs.Mollweide())

# CS = plt.contour(lon, lat, iews, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
#                   linewidths=0.5,colors='k',zorder=1)
# fmt = {} 
# strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 iews 1979-2020 std",fontdict = gfont,pad=10)
cf = plt.pcolormesh(lon,lat,iews,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=mss,zorder=0)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_iews,extend='max',pad=0.1)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_iews,extend='max',pad=0.1)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
cbar.set_label('Instantaneous easthward turbulent surface stress (N m**-2)')
# if not (os.path.exists(ldir0 + 'u10n_full_std_SA.png')):
#     plt.savefig(ldir0 + 'u10n_full_std_SA.png',bbox_inches='tight')
plt.show()































####################################################
#Plots for the Ilhas region


from matplotlib.patches import Circle
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/'
fl = ldir0 + 'full_ERA_wind_means_stds.nc'
ds = xr.open_dataset(fl)
ds = ds.reindex(lat=sorted(ds.lat))



#archipelago:00°55′1″N 29°20′45″W or 0.91694444 N, 29.34583333 W
lat_SPSP_S = 0.8
    #lon_SPSP_S -55 to 15
lat_SPSP_N = 1.2

lat_5N = 5
    #lon_5N -55 to -5
lat_5S = -5
    #lon_5N -40 to 15

# adt_hov = adt.sel(latitude=5.125, longitude=slice(305,355))

lon = ds.lon.values
lat = ds.lat.values
data_u10n_mean = ds.u10n_mean.values
data_u10n_mean, lon = shiftgrid(180., data_u10n_mean, lon, start=False)

u10n_ilhas1 = ds.u10n_mean.sel(lat=slice(-10,10),lon=slice(300,359.75))
u10n_ilhas2 = ds.u10n_mean.sel(lat=slice(-10,10),lon=slice(0,15))
u10n_ilhas= xr.concat([u10n_ilhas1,u10n_ilhas2],dim='lon')
z = u10n_ilhas.values



u10_ilhas = ds.u10_mean.sel(lat=slice(-10,10),lon=slice(-60,15))
v10_ilhas = ds.v10_mean.sel(lat=slice(-10,10),lon=slice(-60,15))

    #for the levels
MinCont_var = np.round(np.nanmin(z),decimals=2)
MaxCont_var = np.round(np.nanmax(z),decimals=2)
levels_var = np.linspace(MinCont_var, MaxCont_var)
levels2_var = np.linspace(MinCont_var, MaxCont_var,6)
ticks_var = np.linspace(MinCont_var, MaxCont_var,6)

lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)

mu = np.nanmean(z)
sigma = np.nanstd(z)
u10n_min = np.nanmin(z)
u10n_max =np.nanmax(z)

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n_mean = [sm,m,ms]

fig = plt.figure(figsize=(7, 5),dpi= 300)
ax1 = plt.axes(projection=ccrs.PlateCarree())
# ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-60.01, 15.01, -10.05, 10.05], ccrs.PlateCarree(central_longitude=0))
# ax1.set_yticks([-10, -5, 0, 5, 10]); ax1.set_yticklabels(y_tick_labels)
# ax1.set_xticks([-60, -45, -30, -15, 0, 15]); ax1.set_xticklabels(x_tick_labels)

cf = plt.pcolormesh(lon2d, lat2d, z,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.delta,zorder=0,
                    vmin=ssm,vmax=mss)
ax1.coastlines(resolution='50m', color='black', linewidth=0.25)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=3)
ax1.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=2)
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax1.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax1.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax1.add_feature(cartopy.feature.OCEAN)
plt.title("ERA5 Neutral Zonal Wind 1979-2020 mean",pad = 20,fontsize=14)
gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                  ylocs=np.arange(-10, 11, 5),
                  linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
# ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
gl.ypadding = 10
gl.xpadding = 10
# Add archipelago
patches = [Circle((-29.34583333, 0.91694444), radius=0.35, color='black')]
for p in patches:
    ax1.add_patch(p)
cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_u10n_mean,extend='both')
cbar.set_label('u10n (m/s)',fontsize=12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.2f}', f'$\mu$ = {m:.2f}',
                          f'$\mu$+$\sigma$ = {ms:.2f}'])
# cbar.ax.text(u10n_min-0.7,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(u10n_max+0.7,0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
# CS = ax1.contour(lon2d, lat2d, z, transform=ccrs.PlateCarree(),levels=[sm,m,ms],
#                   linewidths=0.35,colors='k',zorder=1,inline=True)

# fmt = {} 
# strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax1.clabel(CS, CS.levels, inline=True,fontsize=9,fmt=fmt)
if not (os.path.exists(ldir0 + 'ilhas_full_u10n_mean2.png')):
    plt.savefig(ldir0 + 'ilhas_full_u10n_mean2.png',bbox_inches='tight')




#v10n
v10n_ilhas1 = ds.v10n_mean.sel(lat=slice(-10,10),lon=slice(300,359.75))
v10n_ilhas2 = ds.v10n_mean.sel(lat=slice(-10,10),lon=slice(0,15))
v10n_ilhas= xr.concat([v10n_ilhas1,v10n_ilhas2],dim='lon')
z = v10n_ilhas.values


    #for the levels
MinCont_var = np.round(np.nanmin(z),decimals=2)
MaxCont_var = np.round(np.nanmax(z),decimals=2)
levels_var = np.linspace(MinCont_var, MaxCont_var)
levels2_var = np.linspace(MinCont_var, MaxCont_var,6)
ticks_var = np.linspace(MinCont_var, MaxCont_var,6)

lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)

mu = np.nanmean(z)
sigma = np.nanstd(z)
u10n_min = np.nanmin(z)
u10n_max =np.nanmax(z)

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n_mean = [sm,m,ms]

fig = plt.figure(figsize=(7, 5),dpi= 300)
ax1 = plt.axes(projection=ccrs.PlateCarree())
# ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-60.01, 15.01, -10.05, 10.05], ccrs.PlateCarree(central_longitude=0))
# ax1.set_yticks([-10, -5, 0, 5, 10]); ax1.set_yticklabels(y_tick_labels)
# ax1.set_xticks([-60, -45, -30, -15, 0, 15]); ax1.set_xticklabels(x_tick_labels)

cf = plt.pcolormesh(lon2d, lat2d, z,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.delta,zorder=0)
ax1.coastlines(resolution='50m', color='black', linewidth=0.25)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=3)
ax1.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=2)
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax1.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax1.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax1.add_feature(cartopy.feature.OCEAN)
plt.title("ERA5 Neutral Meridional Wind 1979-2020 mean",pad = 20,fontsize=14)
gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                  ylocs=np.arange(-10, 11, 5),
                  linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
# ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
gl.ypadding = 10
gl.xpadding = 10
# Add archipelago
patches = [Circle((-29.34583333, 0.91694444), radius=0.35, color='black')]
for p in patches:
    ax1.add_patch(p)
cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_u10n_mean,extend='both')
cbar.set_label('v10n (m/s)',fontsize=12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.2f}', f'$\mu$ = {m:.2f}',
                          f'$\mu$+$\sigma$ = {ms:.2f}'])
# cbar.ax.text(u10n_min-0.7,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(u10n_max+0.7,0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
# CS = ax1.contour(lon2d, lat2d, z, transform=ccrs.PlateCarree(),levels=[sm,m,ms],
#                   linewidths=0.35,colors='k',zorder=1,inline=True)

# fmt = {} 
# strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax1.clabel(CS, CS.levels, inline=True,fontsize=9,fmt=fmt)
if not (os.path.exists(ldir0 + 'ilhas_full_v10n_mean.png')):
    plt.savefig(ldir0 + 'ilhas_full_v10n_mean.png',bbox_inches='tight')


#u10
u10_ilhas1 = ds.u10_mean.sel(lat=slice(-10,10),lon=slice(300,359.75))
u10_ilhas2 = ds.u10_mean.sel(lat=slice(-10,10),lon=slice(0,15))
u10_ilhas= xr.concat([u10_ilhas1,u10_ilhas2],dim='lon')
z = u10_ilhas.values


    #for the levels
MinCont_var = np.round(np.nanmin(z),decimals=2)
MaxCont_var = np.round(np.nanmax(z),decimals=2)
levels_var = np.linspace(MinCont_var, MaxCont_var)
levels2_var = np.linspace(MinCont_var, MaxCont_var,6)
ticks_var = np.linspace(MinCont_var, MaxCont_var,6)

lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)

mu = np.nanmean(z)
sigma = np.nanstd(z)
u10n_min = np.nanmin(z)
u10n_max =np.nanmax(z)

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n_mean = [sm,m,ms]

fig = plt.figure(figsize=(7, 5),dpi= 300)
ax1 = plt.axes(projection=ccrs.PlateCarree())
# ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-60.01, 15.01, -10.05, 10.05], ccrs.PlateCarree(central_longitude=0))
# ax1.set_yticks([-10, -5, 0, 5, 10]); ax1.set_yticklabels(y_tick_labels)
# ax1.set_xticks([-60, -45, -30, -15, 0, 15]); ax1.set_xticklabels(x_tick_labels)

cf = plt.pcolormesh(lon2d, lat2d, z,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.delta,zorder=0)
ax1.coastlines(resolution='50m', color='black', linewidth=0.25)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=3)
ax1.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=2)
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax1.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax1.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax1.add_feature(cartopy.feature.OCEAN)
plt.title("ERA5 Zonal Wind 1979-2020 mean",pad = 20,fontsize=14)
gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                  ylocs=np.arange(-10, 11, 5),
                  linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
# ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
gl.ypadding = 10
gl.xpadding = 10
# Add archipelago
patches = [Circle((-29.34583333, 0.91694444), radius=0.35, color='black')]
for p in patches:
    ax1.add_patch(p)
cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_u10n_mean,extend='both')
cbar.set_label('u10 (m/s)',fontsize=12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.2f}', f'$\mu$ = {m:.2f}',
                          f'$\mu$+$\sigma$ = {ms:.2f}'])
# cbar.ax.text(u10n_min-0.7,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(u10n_max+0.7,0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
CS = ax1.contour(lon2d, lat2d, z, transform=ccrs.PlateCarree(),levels=[sm,m,ms],
                  linewidths=0.35,colors='k',zorder=1,inline=True)

fmt = {} 
strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax1.clabel(CS, CS.levels, inline=True,fontsize=9,fmt=fmt)
if not (os.path.exists(ldir0 + 'ilhas_full_u10_mean.png')):
    plt.savefig(ldir0 + 'ilhas_full_u10_mean.png',bbox_inches='tight')


#v10
v10_ilhas1 = ds.v10_mean.sel(lat=slice(-10,10),lon=slice(300,359.75))
v10_ilhas2 = ds.v10_mean.sel(lat=slice(-10,10),lon=slice(0,15))
v10_ilhas= xr.concat([v10_ilhas1,v10_ilhas2],dim='lon')
z = v10_ilhas.values


    #for the levels
MinCont_var = np.round(np.nanmin(z),decimals=2)
MaxCont_var = np.round(np.nanmax(z),decimals=2)
levels_var = np.linspace(MinCont_var, MaxCont_var)
levels2_var = np.linspace(MinCont_var, MaxCont_var,6)
ticks_var = np.linspace(MinCont_var, MaxCont_var,6)

lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)

mu = np.nanmean(z)
sigma = np.nanstd(z)
u10n_min = np.nanmin(z)
u10n_max =np.nanmax(z)

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n_mean = [sm,m,ms]

fig = plt.figure(figsize=(7, 5),dpi= 300)
ax1 = plt.axes(projection=ccrs.PlateCarree())
# ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-60.01, 15.01, -10.05, 10.05], ccrs.PlateCarree(central_longitude=0))
# ax1.set_yticks([-10, -5, 0, 5, 10]); ax1.set_yticklabels(y_tick_labels)
# ax1.set_xticks([-60, -45, -30, -15, 0, 15]); ax1.set_xticklabels(x_tick_labels)

cf = plt.pcolormesh(lon2d, lat2d, z,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.delta,zorder=0)
ax1.coastlines(resolution='50m', color='black', linewidth=0.25)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=3)
ax1.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=2)
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax1.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax1.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax1.add_feature(cartopy.feature.OCEAN)
plt.title("ERA5 Neutral Meridional Wind 1979-2020 mean",pad = 20,fontsize=14)
gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                  ylocs=np.arange(-10, 11, 5),
                  linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
# ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
gl.ypadding = 10
gl.xpadding = 10
# Add archipelago
patches = [Circle((-29.34583333, 0.91694444), radius=0.35, color='black')]
for p in patches:
    ax1.add_patch(p)
cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_u10n_mean,extend='both')
cbar.set_label('v10 (m/s)',fontsize=12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.2f}', f'$\mu$ = {m:.2f}',
                          f'$\mu$+$\sigma$ = {ms:.2f}'])
# cbar.ax.text(u10n_min-0.7,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
# cbar.ax.text(u10n_max+0.7,0,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
# CS = ax1.contour(lon2d, lat2d, z, transform=ccrs.PlateCarree(),levels=[sm,m,ms],
#                   linewidths=0.35,colors='k',zorder=1,inline=True)

# fmt = {} 
# strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
# for l, s in zip(CS.levels, strs): 
#     fmt[l] = s
# ax1.clabel(CS, CS.levels, inline=True,fontsize=9,fmt=fmt)
if not (os.path.exists(ldir0 + 'ilhas_full_v10_mean.png')):
    plt.savefig(ldir0 + 'ilhas_full_v10_mean.png',bbox_inches='tight')
























#now for the std
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/CMC_GHRSST/'
fl = ldir0 + 'full_std.nc'
ds = xr.open_dataset(fl)

#archipelago:00°55′1″N 29°20′45″W or 0.91694444 N, 29.34583333 W
lat_SPSP_S = 0.8
    #lon_SPSP_S -55 to 15
lat_SPSP_N = 1.2

lat_5N = 5
    #lon_5N -55 to -5
lat_5S = -5
    #lon_5N -40 to 15

# adt_hov = adt.sel(latitude=5.125, longitude=slice(305,355))
sst_ilhas = ds.sst_std.sel(lat=slice(-10,10),lon=slice(-60,15))
z = sst_ilhas.values

# adt_hov_num = adt_hov.values
# lon= adt_hov.longitude.values
# lons=np.arange(1,np.size(lon))


### PLOTTING from scratch ###

    #for the levels
MinCont_var = np.round(np.nanmin(z),decimals=2)
MaxCont_var = np.round(np.nanmax(z),decimals=2)
levels_var = np.linspace(MinCont_var, MaxCont_var)
levels2_var = np.linspace(MinCont_var, MaxCont_var,6)
ticks_var = np.linspace(MinCont_var, MaxCont_var,6)

lon = sst_ilhas.lon.values
lat = sst_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)

mu = np.nanmean(z)
sigma = np.nanstd(z)
sst_min = np.nanmin(z)
sst_max =np.nanmax(z)

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_sst_mean = [sm,m,ms,mss]

fig = plt.figure(figsize=(7, 5),dpi= 300)

ax1 = plt.axes(projection=ccrs.PlateCarree())
# ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-60.01, 15.01, -10.05, 10.05], ccrs.PlateCarree(central_longitude=0))
# ax1.set_yticks([-10, -5, 0, 5, 10]); ax1.set_yticklabels(y_tick_labels)
# ax1.set_xticks([-60, -45, -30, -15, 0, 15]); ax1.set_xticklabels(x_tick_labels)
ax1.coastlines(resolution='50m', color='black', linewidth=0.25)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=2)
ax1.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=3)
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax1.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax1.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax1.add_feature(cartopy.feature.OCEAN)
cf = plt.pcolormesh(lon2d, lat2d, z,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.amp,vmin=0)

plt.title("CMC-GHRSST 'Analysed SST' 1991-2017 std",pad = 20,fontsize=14)
gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                  ylocs=np.arange(-10, 11, 5),
                  linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
# ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
gl.ypadding = 10
gl.xpadding = 10
# Add archipelago
patches = [Circle((-29.34583333, 0.91694444), radius=0.35, color='black')]
for p in patches:
    ax1.add_patch(p)
cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_sst_mean,extend='max')
cbar.set_label('Analysed SST (\u00B0C)',fontsize=12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.2f}', f'$\mu$ = {m:.2f}',
                          f'$\mu$+$\sigma$ = {ms:.2f}',f'$\mu$+2$\sigma$ = {mss:.2f}'])
# cbar.ax.text(-0.5,0,f'MIN\n{sst_min:.2f}',ha='center',va='center')
# cbar.ax.text(3.5+0.5,0,f'MAX\n{sst_max:.2f}',ha='center',va='center')
CS = ax1.contour(lon, lat, z, transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                 linewidths=0.35,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', 'ds = xr.Dataset.from_dict(ddd)
ds.to_netcdf('full_stds2.nc', format='NETCDF4',
             encoding=encoding)
$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
if not (os.path.exists(ldir0 + 'ilhas_full_sst_std.png')):
    plt.savefig(ldir0 + 'ilhas_full_sst_std.png',bbox_inches='tight')







































#we need a plot of full std
ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/"
ldir1 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/"

path = 'full_wind_means_stds.nc'

ds = xr.open_dataset(ldir0+path)

# u10n std
lon = ds.u10n_std.lon.values
lat = ds.u10n_std.lat.values
u10n = ds.u10n_std.values

u10n, lon = shiftgrid(180., u10n, lon, start=False)

mu = np.nanmean(ds.u10n_std.values)
sigma = np.nanstd(ds.u10n_std.values)
u10n_min = np.nanmin(ds.u10n_std.values)
u10n_max =np.nanmax(ds.u10n_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_min:.2f}, max is {u10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10n = [0,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

CS = plt.contour(lon, lat, u10n, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
                 linewidths=0.25,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$', '$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10n 1979-2020 Standard Deviation",fontsize=18,pad=10)
cf = plt.pcolormesh(lon,lat,u10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=4,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
# gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",pad=0.1,extend='max')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10n,extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(0.2,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
cbar.ax.text(4.5,0.6,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral zonal wind at 10m (m/s)',fontsize=12)
# if not (os.path.exists(ldir0 + 'u10n_full_std.png')):
#     plt.savefig(ldir0 + 'u10n_full_std.png',bbox_inches='tight')
plt.show()




# v10n std
lon = ds.v10n_std.lon.values
lat = ds.v10n_std.lat.values
v10n = ds.v10n_std.values

v10n, lon = shiftgrid(180., v10n, lon, start=False)

mu = np.nanmean(ds.v10n_std.values)
sigma = np.nanstd(ds.v10n_std.values)
v10n_min = np.nanmin(ds.v10n_std.values)
v10n_max =np.nanmax(ds.v10n_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10n = [0,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

CS = plt.contour(lon, lat, v10n, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
                 linewidths=0.25,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$', '$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10n 1979-2020 Standard Deviation",fontsize=18,pad=10)
cf = plt.pcolormesh(lon,lat,v10n,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=4,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
# gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",pad=0.1,extend='max')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10n,extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(0.2,0,f'MIN\n{v10n_min:.2f}',ha='center',va='center')
cbar.ax.text(4.5,0.6,f'MAX\n{v10n_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral meridional wind at 10m (m/s)',fontsize=12)
if not (os.path.exists(ldir0 + 'v10n_full_std.png')):
    plt.savefig(ldir0 + 'v10n_full_std.png',bbox_inches='tight')
plt.show()



# u10 std
lon = ds.u10_std.lon.values
lat = ds.u10_std.lat.values
u10 = ds.u10_std.values

u10, lon = shiftgrid(180., u10, lon, start=False)

mu = np.nanmean(ds.u10_std.values)
sigma = np.nanstd(ds.u10_std.values)
u10_min = np.nanmin(ds.u10_std.values)
u10_max =np.nanmax(ds.u10_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_min:.2f}, max is {u10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_u10 = [0,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

CS = plt.contour(lon, lat, u10, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
                 linewidths=0.25,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$', '$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 u10 1979-2020 Standard Deviation",fontsize=18,pad=10)
cf = plt.pcolormesh(lon,lat,u10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=4,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
# gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",pad=0.1,extend='max')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10,extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(0.2,0,f'MIN\n{u10_min:.2f}',ha='center',va='center')
cbar.ax.text(4.5,0.6,f'MAX\n{u10_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral meridional wind at 10m (m/s)',fontsize=12)
if not (os.path.exists(ldir0 + 'u10_full_std.png')):
    plt.savefig(ldir0 + 'u10_full_std.png',bbox_inches='tight')
plt.show()



# v10 std
lon = ds.v10_std.lon.values
lat = ds.v10_std.lat.values
v10 = ds.v10_std.values

v10, lon = shiftgrid(180., v10, lon, start=False)

mu = np.nanmean(ds.v10_std.values)
sigma = np.nanstd(ds.v10_std.values)
v10_min = np.nanmin(ds.v10_std.values)
v10_max =np.nanmax(ds.v10_std.values)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_min:.2f}, max is {v10_max:.2f}')


[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
print([ssm,sm,m,ms,mss])

#for the colorbar levels
ticks_v10 = [0,sm,m,ms,mss]
# gfont = {'fontname':'Helvetica','fontsize' : 20}

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())

CS = plt.contour(lon, lat, v10, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
                 linewidths=0.25,colors='k',zorder=1,inline=True)
fmt = {} 
strs = ['$\mu$', '$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1,zorder=2)
ax.add_feature(cfeature.LAND, zorder=3,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4)
ax.add_feature(cfeature.BORDERS, linewidths=0.25,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.set_global()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
plt.title("ERA5 v10 1979-2020 Standard Deviation",fontsize=18,pad=10)
cf = plt.pcolormesh(lon,lat,v10,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
                    vmin=0,vmax=4,zorder=0)
gl = ax.gridlines(draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
# gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",pad=0.1,extend='max')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10,extend='max')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['0','$\mu$-$\sigma$','$\mu$','$\mu$+$\sigma$','2$\mu$+$\sigma$']) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
# cbar.ax.text(0.2,0,f'MIN\n{v10_min:.2f}',ha='center',va='center')
cbar.ax.text(4.5,0.6,f'MAX\n{v10_max:.2f}',ha='center',va='center')
cbar.set_label('Neutral meridional wind at 10m (m/s)',fontsize=12)
if not (os.path.exists(ldir0 + 'v10_full_std.png')):
    plt.savefig(ldir0 + 'v10_full_std.png',bbox_inches='tight')
plt.show()


#######################################################################################
#01/25 - New plots for Ilhas

path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/full_wind_means_stds.nc"
ds = xr.open_dataset(path2)

vari = ds.v10n_mean #12 x 720 x 1440
# u10n MEAN
lon = ds.u10n_mean.lon.values
lat = ds.u10n_mean.lat.values

#u10n
#lets plot for the Archipelago
latN = 10.25
latS = -10.25
lonW = 295.25
lonE = 15.25

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#extract lat and lons
lats = vari_mean_ilhas.lat.values
lons = vari_mean_ilhas.lon.values

mu, sigma, vari_min, vari_max = stats(vari)

# mu = np.nanmean(u10n) #0.4754307347234847
# sigma = np.nanstd(u10n) #0.12794454349812165
# u10n_min = np.nanmin(u10n)
# u10n_max = np.nanmax(u10n)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]

# print([ssm,sm,m,ms,mss])


#for the colorbar levels
ticks = [sm,m,ms]
ticks_alt = [-3,-2,-1,0,1,2,3,4]
gfont = {'fontsize' : 16}

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


vals = np.array([[-3., 0], [0, 4]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

#Adapted for u10n
fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.5)
ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
CS = plt.contour(lons, lats, vari, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                  levels=[sm,m,ms],zorder=1)
fmt = {} 
strs = ['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
plt.title("ERA5 v10n 1979-2020 mean",fontdict = gfont)
cf = plt.pcolormesh(lons,lats,vari,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=-3,vmax=4,zorder=0,norm=norm)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_alt,extend='both',pad=0.1,shrink=0.9)
cbar.set_label('Neutral Meridional wind at 10m (m/s)',fontsize = 12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks,extend='both',pad=0.1,shrink=0.9)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$']) 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
gl.xlabels_top = False
gl.ylabels_right = False
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
cbar.ax.get_yaxis().set_ticks([])
text(0, -0.8, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
text(1, -0.8, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
# if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
#     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
plt.show()


#we also need stress
path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/full_ERA_stress_means_stds.nc"
ds = xr.open_dataset(path2)

vari = ds.inss_mean #12 x 720 x 1440
# u10n MEAN
lon = ds.iews_mean.lon.values
lat = ds.iews_mean.lat.values

#u10n
#lets plot for the Archipelago
latN = 10.25
latS = -10.25
lonW = 295.25
lonE = 15.25

vari_mean1 = vari.sel(lat=slice(latN,latS), lon=slice(lonW,359.75))
vari_mean2 = vari.sel(lat=slice(latN,latS), lon=slice(0,lonE))
vari_mean_ilhas = xr.concat([vari_mean1, vari_mean2], dim='lon')
vari = vari_mean_ilhas.values #new vari!!!!

#extract lat and lons
lats = vari_mean_ilhas.lat.values
lons = vari_mean_ilhas.lon.values

mu, sigma, vari_min, vari_max = stats(vari)

# mu = np.nanmean(u10n) #0.4754307347234847
# sigma = np.nanstd(u10n) #0.12794454349812165
# u10n_min = np.nanmin(u10n)
# u10n_max = np.nanmax(u10n)
# print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')

[ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]

# print([ssm,sm,m,ms,mss])


#for the colorbar levels
ticks = [ssm,sm,m,ms,mss]
gfont = {'fontsize' : 16}

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


vals = np.array([[-3., 0], [0, 4]]) 
vmin = vals.min()
vmax = vals.max()

norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

#Adapted for u10n
fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
arch = ax.plot(-29.34583333,0.91694444, 'k*', markersize=3, markeredgewidth = 0.25, transform=ccrs.PlateCarree()) # WHY
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.1)
ax.add_feature(cfeature.LAND, zorder=1,edgecolor='black',linewidths=0.25)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.25)
ax.add_feature(cfeature.RIVERS, linewidths=0.5)
ax.set_extent([-65, 15, -10.1, 10.1], crs=ccrs.PlateCarree())
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidths=0.25)
CS = plt.contour(lons, lats, vari, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
                  levels=[ssm,sm,m,ms,mss],zorder=1)
fmt = {} 
strs = ['$\mu$-2$\sigma$','$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
for l, s in zip(CS.levels, strs): 
    fmt[l] = s
ax.clabel(CS, CS.levels, inline=True,fontsize=5,fmt=fmt)
plt.title("ERA5 INSS 1979-2020 mean",fontdict = gfont)
cf = plt.pcolormesh(lons,lats,vari,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                    vmin=ssm,vmax=mss,zorder=0,norm=norm)
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks,extend='both',pad=0.1,shrink=0.9)
cbar.set_label('Instantaneous Northward Turbulent Surface Stress ($N m^{-2}$)',fontsize = 12)
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks,extend='both',pad=0.1,shrink=0.9)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$','$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=0.25, color='black', alpha=0.5, linestyle='dashed')
gl.xlabels_top = False
gl.ylabels_right = False
# gl.xlines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-10,-5,0,5,10])
cbar.ax.get_yaxis().set_ticks([])
text(0, -0.8, f'MIN = {vari_min:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
text(1, -0.8, f'MAX = {vari_max:.2f}', fontsize=12,ha='center', va='center', transform=ax.transAxes)
# if not (os.path.exists(ldir1 + 'u10n_full_mean_ilhas.png')):
#     plt.savefig(ldir1 + 'u10n_full_mean_ilhas.png',bbox_inches='tight')
plt.show()












