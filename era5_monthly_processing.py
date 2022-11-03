#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:15:35 2020

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
import warnings
import matplotlib.cbook
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import cmocean
from dask.diagnostics import ProgressBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
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


ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/download.nc"
ds = xr.open_dataset(ldir0)

u10n = ds.u10n.sel(expver=1)
v10n = ds.v10n.sel(expver=1)
u10 = ds.u10.sel(expver=1)
v10 = ds.v10.sel(expver=1)


from dask.diagnostics import ProgressBar
from dask.distributed import Client
client = Client() #ok lets try this
client

with ProgressBar():
    u10n_mean_months = u10n.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 


v10n_mean_months = v10n.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
u10_mean_months = u10.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
v10_mean_months = v10.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 


u10n_std_months = u10n.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
v10n_std_months = v10n.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
u10_std_months = u10.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
v10_std_months = v10.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load()



#Create a neat dataset
lat = u10n_mean_months.latitude.values
lon = u10n_mean_months.longitude.values
month = u10n_mean_months.month.values
ddd2 = {'month':{'dims': 'month','data': month, 'attrs': {'units': 'none'}},
        'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N'}},
       'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E'}}}

from collections import OrderedDict as od
z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['u10n_mean_months'] = u10n_mean_months.values
z_c1['v10n_mean_months'] = v10n_mean_months.values
z_c2['u10_mean_months'] = u10_mean_months.values
z_c2['v10_mean_months'] = v10_mean_months.values
z_c3['u10n_std_months'] = u10n_std_months.values
z_c3['v10n_std_months'] = v10n_std_months.values
z_c4['u10_std_months'] = u10_std_months.values
z_c4['v10_std_months'] = v10_std_months.values


encoding = {}
for key in z_c1.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c1[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c2.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c2[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
for key in z_c3.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c3[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c4.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c4[key],
                          'attrs': {'units': 'm/s'}}})
    encoding.update({key: {'dtype': 'float32', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
ds = xr.Dataset.from_dict(ddd2)
ds.to_netcdf('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly_means_stds.nc', format='NETCDF4',
             encoding=encoding)


###### PLOTTING

#check
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/'
ds = xr.open_dataset(ldir0 + 'monthly_means_stds.nc')

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

u10n_mean = ds.u10n_mean_months
v10n_mean = ds.v10n_mean_months
u10_mean = ds.u10_mean_months
v10_mean = ds.v10_mean_months


lats = u10n_mean[0].lat.values
lons = u10n_mean[0].lon.values

# for u10n
z = u10n_mean.values

u10n = {}
namespace = globals()
u10n_list=[]
for m in np.arange(0,12):
    u10n[month[m]] = u10n_mean[m,:,:].values
    u10n_list.append(u10n_mean[m,:,:].values) #this works for separating the months
    namespace[f'u10n_mean_{month[m]}'] = u10n_mean[m] #separates the 12 dataarrays by name
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
ticks_u10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in u10n.keys():
    lons = u10n_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = u10n[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Mollweide())
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
    plt.title(f'ERA5 u10n 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10n,extend='both',pad=0.1)
    cbar.set_label('Neutral zonal wind at 10m (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10n,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.7,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.7,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir0 + f'u10n_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'u10n_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()



# for v10n
z = v10n_mean.values

v10n = {}
namespace = globals()
v10n_list=[]
for m in np.arange(0,12):
    v10n[month[m]] = v10n_mean[m,:,:].values
    v10n_list.append(v10n_mean[m,:,:].values) #this works for separating the months
    namespace[f'v10n_mean_{month[m]}'] = v10n_mean[m] #separates the 12 dataarrays by name
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
ticks_v10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in v10n.keys():
    lons = v10n_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = v10n[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Mollweide())
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
    plt.title(f'ERA5 v10n 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_v10n,extend='both',pad=0.1)
    cbar.set_label('Neutral meridional wind at 10m (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_v10n,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.7,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.7,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir0 + f'v10n_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'v10n_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()

#create movie


#check
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/'
ds = xr.open_dataset(ldir0 + 'monthly_means_stds.nc')

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

u10n_mean = ds.u10n_mean_months
v10n_mean = ds.v10n_mean_months
u10_mean = ds.u10_mean_months
v10_mean = ds.v10_mean_months


lats = u10n_mean[0].lat.values
lons = u10n_mean[0].lon.values

# for u10n
z = u10n_mean.values

u10n = {}
namespace = globals()
u10n_list=[]
for m in np.arange(0,12):
    u10n[month[m]] = u10n_mean[m,:,:].values
    u10n_list.append(u10n_mean[m,:,:].values) #this works for separating the months
    namespace[f'u10n_mean_{month[m]}'] = u10n_mean[m] #separates the 12 dataarrays by name
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
ticks_u10n = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in u10n.keys():
    lons = u10n_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = u10n[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Mollweide())
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
    plt.title(f'ERA5 u10n 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10n,extend='both',pad=0.1)
    cbar.set_label('Neutral zonal wind at 10m (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10n,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.7,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.7,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir0 + f'u10n_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'u10n_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()









#now for the ilhas region
from matplotlib.patches import Circle
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly/'
fl = ldir0 + 'monthly_means_stds.nc'
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
u10n_ilhas1 = ds.u10n_mean_months.sel(lat=slice(-10,10),lon=slice(300,359.75))
u10n_ilhas2 = ds.u10n_mean_months.sel(lat=slice(-10,10),lon=slice(0,15))
u10n_ilhas = xr.concat([u10n_ilhas1,u10n_ilhas2],dim='lon')

#plotting monthly means
lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)


u10n = {}
namespace = globals()
sst_list=[]
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    u10n[month[m-1]] = u10n_ilhas.sel(month=m)
print('this worked')


ticks_sst_mean = [sm,m,ms]
ticks_up=np.arange(0,31,step=5)
gfont = {'fontsize' : 16}



n=0
for mon in u10n.keys():
    mu = round(np.nanmean(u10n[mon]),2)
    sigma = round(np.nanstd(u10n[mon]),2)
    u10n_min = round(np.nanmin(u10n[mon]),2)
    u10n_max = round(np.nanmax(u10n[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_min:.2f}, max is {u10n_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_u10n_mean = [sm,m,ms]
    # lons = u10n_ilhas.lon
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
    cf = plt.pcolormesh(lon2d, lat2d, u10n_ilhas[n],transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmocean.cm.delta)
    plt.title(f"ERA5 Neutral Zonal Wind 1979-2020 {mon} mean",fontdict = gfont,pad = 20)
    gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                      ylocs=np.arange(-10, 11, 5),
                      linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    gl.ypadding = 10
    gl.xpadding = 10
    # Add archipelago
    patches = [Circle((-29.34583333, 0.91694444), radius=0.25, color='red')]
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
    cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.1f}', f'$\mu$ = {m:.1f}',
                              f'$\mu$+$\sigma$ = {ms:.1f}'],fontsize=12)
    # cbar.ax.text(22.2,27,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
    # cbar.ax.text(31.8,27,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
    # CS = ax1.contour(lon, lat, u10n[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
    #                  linewidths=0.35,colors='k',zorder=1,inline=True)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #     fmt[l] = s
    # ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
    if not (os.path.exists(ldir0 + f'ilhas_monthly_u10n_{mon}_mean.png')):
        plt.savefig(ldir0 + f'ilhas_monthly_u10n_{mon}_mean.png',bbox_inches='tight')
    n=n+1



#v10n
# adt_hov = adt.sel(latitude=5.125, longitude=slice(305,355))
v10n_ilhas1 = ds.v10n_mean_months.sel(lat=slice(-10,10),lon=slice(300,359.75))
v10n_ilhas2 = ds.v10n_mean_months.sel(lat=slice(-10,10),lon=slice(0,15))
v10n_ilhas = xr.concat([v10n_ilhas1,v10n_ilhas2],dim='lon')

#plotting monthly means
lon = v10n_ilhas.lon.values
lat = v10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)


v10n = {}
namespace = globals()
sst_list=[]
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    v10n[month[m-1]] = v10n_ilhas.sel(month=m)
print('this worked')


ticks_sst_mean = [sm,m,ms]
ticks_up=np.arange(0,31,step=5)
gfont = {'fontsize' : 16}



n=0
for mon in v10n.keys():
    mu = round(np.nanmean(v10n[mon]),2)
    sigma = round(np.nanstd(v10n[mon]),2)
    v10n_min = round(np.nanmin(v10n[mon]),2)
    v10n_max = round(np.nanmax(v10n[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_min:.2f}, max is {v10n_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_v10n_mean = [sm,m,ms]
    # lons = v10n_ilhas.lon
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
    cf = plt.pcolormesh(lon2d, lat2d, v10n_ilhas[n],transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmocean.cm.delta)
    plt.title(f"ERA5 Neutral Meridional Wind 1979-2020 {mon} mean",fontdict = gfont,pad = 20)
    gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                      ylocs=np.arange(-10, 11, 5),
                      linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    gl.ypadding = 10
    gl.xpadding = 10
    # Add archipelago
    patches = [Circle((-29.34583333, 0.91694444), radius=0.25, color='red')]
    for p in patches:
        ax1.add_patch(p)
    cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_v10n_mean,extend='both')
    cbar.set_label('v10n (m/s)',fontsize=12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.1f}', f'$\mu$ = {m:.1f}',
                              f'$\mu$+$\sigma$ = {ms:.1f}'],fontsize=12)
    # cbar.ax.text(22.2,27,f'MIN\n{v10n_min:.2f}',ha='center',va='center')
    # cbar.ax.text(31.8,27,f'MAX\n{v10n_max:.2f}',ha='center',va='center')
    # CS = ax1.contour(lon, lat, v10n[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
    #                  linewidths=0.35,colors='k',zorder=1,inline=True)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #     fmt[l] = s
    # ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
    if not (os.path.exists(ldir0 + f'ilhas_monthly_v10n_{mon}_mean.png')):
        plt.savefig(ldir0 + f'ilhas_monthly_v10n_{mon}_mean.png',bbox_inches='tight')
    n=n+1



#u10
# adt_hov = adt.sel(latitude=5.125, longitude=slice(305,355))
u10_ilhas1 = ds.u10_mean_months.sel(lat=slice(-10,10),lon=slice(300,359.75))
u10_ilhas2 = ds.u10_mean_months.sel(lat=slice(-10,10),lon=slice(0,15))
u10_ilhas = xr.concat([u10_ilhas1,u10_ilhas2],dim='lon')

#plotting monthly means
lon = u10_ilhas.lon.values
lat = u10_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)


u10 = {}
namespace = globals()
sst_list=[]
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    u10[month[m-1]] = u10_ilhas.sel(month=m)
print('this worked')


ticks_sst_mean = [sm,m,ms]
ticks_up=np.arange(0,31,step=5)
gfont = {'fontsize' : 16}



n=0
for mon in u10.keys():
    mu = round(np.nanmean(u10[mon]),2)
    sigma = round(np.nanstd(u10[mon]),2)
    u10_min = round(np.nanmin(u10[mon]),2)
    u10_max = round(np.nanmax(u10[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_min:.2f}, max is {u10_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_u10_mean = [sm,m,ms]
    # lons = u10_ilhas.lon
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
    cf = plt.pcolormesh(lon2d, lat2d, u10_ilhas[n],transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmocean.cm.delta)
    plt.title(f"ERA5 Zonal Wind 1979-2020 {mon} mean",fontdict = gfont,pad = 20)
    gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                      ylocs=np.arange(-10, 11, 5),
                      linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    gl.ypadding = 10
    gl.xpadding = 10
    # Add archipelago
    patches = [Circle((-29.34583333, 0.91694444), radius=0.25, color='red')]
    for p in patches:
        ax1.add_patch(p)
    cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_u10_mean,extend='both')
    cbar.set_label('u10 (m/s)',fontsize=12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.1f}', f'$\mu$ = {m:.1f}',
                              f'$\mu$+$\sigma$ = {ms:.1f}'],fontsize=12)
    # cbar.ax.text(22.2,27,f'MIN\n{u10_min:.2f}',ha='center',va='center')
    # cbar.ax.text(31.8,27,f'MAX\n{u10_max:.2f}',ha='center',va='center')
    # CS = ax1.contour(lon, lat, u10[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
    #                  linewidths=0.35,colors='k',zorder=1,inline=True)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #     fmt[l] = s
    # ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
    if not (os.path.exists(ldir0 + f'ilhas_monthly_u10_{mon}_mean.png')):
        plt.savefig(ldir0 + f'ilhas_monthly_u10_{mon}_mean.png',bbox_inches='tight')
    n=n+1



#v10
# adt_hov = adt.sel(latitude=5.125, longitude=slice(305,355))
v10_ilhas1 = ds.v10_mean_months.sel(lat=slice(-10,10),lon=slice(300,359.75))
v10_ilhas2 = ds.v10_mean_months.sel(lat=slice(-10,10),lon=slice(0,15))
v10_ilhas = xr.concat([v10_ilhas1,v10_ilhas2],dim='lon')

#plotting monthly means
lon = v10_ilhas.lon.values
lat = v10_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)


v10 = {}
namespace = globals()
sst_list=[]
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    v10[month[m-1]] = v10_ilhas.sel(month=m)
print('this worked')


ticks_sst_mean = [sm,m,ms]
ticks_up=np.arange(0,31,step=5)
gfont = {'fontsize' : 16}



n=0
for mon in v10.keys():
    mu = round(np.nanmean(v10[mon]),2)
    sigma = round(np.nanstd(v10[mon]),2)
    v10_min = round(np.nanmin(v10[mon]),2)
    v10_max = round(np.nanmax(v10[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_min:.2f}, max is {v10_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_v10_mean = [sm,m,ms]
    # lons = v10_ilhas.lon
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
    cf = plt.pcolormesh(lon2d, lat2d, v10_ilhas[n],transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmocean.cm.delta)
    plt.title(f"ERA5 Zonal Wind 1979-2020 {mon} mean",fontdict = gfont,pad = 20)
    gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                      ylocs=np.arange(-10, 11, 5),
                      linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    gl.ypadding = 10
    gl.xpadding = 10
    # Add archipelago
    patches = [Circle((-29.34583333, 0.91694444), radius=0.25, color='red')]
    for p in patches:
        ax1.add_patch(p)
    cbar = plt.colorbar(ax=ax1,orientation="horizontal",ticks=ticks_v10_mean,extend='both')
    cbar.set_label('v10 (m/s)',fontsize=12)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.1f}', f'$\mu$ = {m:.1f}',
                              f'$\mu$+$\sigma$ = {ms:.1f}'],fontsize=12)
    # cbar.ax.text(22.2,27,f'MIN\n{v10_min:.2f}',ha='center',va='center')
    # cbar.ax.text(31.8,27,f'MAX\n{v10_max:.2f}',ha='center',va='center')
    # CS = ax1.contour(lon, lat, v10[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
    #                  linewidths=0.35,colors='k',zorder=1,inline=True)
    # fmt = {} 
    # strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    # for l, s in zip(CS.levels, strs): 
    #     fmt[l] = s
    # ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
    if not (os.path.exists(ldir0 + f'ilhas_monthly_v10_{mon}_mean.png')):
        plt.savefig(ldir0 + f'ilhas_monthly_v10_{mon}_mean.png',bbox_inches='tight')
    n=n+1



#now for STDs (this can wait)

ldir0='/media/hbatistuzzo/DATA/Ilhas/Era5/wind/'
ds = xr.open_dataset(ldir0+'monthly_means_stds.nc')

u10n_std = u10n_std_months.values
v10n_std = v10n_std_months.values
u10_std = u10_std_months.values
v10_std = v10_std_months.values



#for sst_mean
lon2d, lat2d = np.meshgrid(lon, lat)



u10n_std = {}
namespace = globals()
u10n_std_list=[]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    u10n_std[month[m-1]] = u10n_std_months.sel(month=m)
    u10n_std[month[m-1]] = u10n_std[month[m-1]].values
    # u10n_std[month[m-1]], lon = shiftgrid(180., u10n[month[m-1]], lon, start=False)
    # sst_list.append(sst.isel(month=m).values) #this works for separating the months
    # namespace[f'sst_mean_{month[m]}'] = ds_mean[m] #separates the 12 dataarrays by name
    print('this worked')

#argh fix the scale
# sst[month[m]] = sst[month[m]].values - 273


# ssm = ssm
# mss = mss
# ticks_u10n_std_mean = [sm,m,ms]
# ticks_up=np.arange(0,31,step=5)
# gfont = {'fontsize' : 16}

n=1
for mon in months:
    lon = ds.u10n_std_months.lon.values
    lat = ds.u10n_std_months.lat.values
    u10n_std = ds.u10n_std_months.sel(month=n).values
    u10n_std, lon = shiftgrid(180., u10n_std, lon, start=False)
    mu = round(np.nanmean(u10n_std),2)
    sigma = round(np.nanstd(u10n_std),2)
    u10n_std_min = round(np.nanmin(u10n_std),2)
    u10n_std_max = round(np.nanmax(u10n_std),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_std_min:.2f}, max is {u10n_std_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_u10n = [0,m,ms,mss]
    # lons = ds_mean.lon.values
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lon, lat, u10n_std, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
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
    plt.title(f"ERA5 u10n 1979-2020 {mon} Standard Deviation",fontsize=18,pad=10)
    cf = plt.pcolormesh(lon,lat,u10n_std,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
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
    cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # cbar.ax.text(0.2,0,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
    # cbar.ax.text(3.3,0.6,f'MAX\n{u10n_std_max:.2f}',ha='center',va='center')
    cbar.set_label('Neutral zonal wind at 10m (m/s)',fontsize=12)
    if not (os.path.exists(ldir0 + f'u10n_std_{mon}_std.png')):
        plt.savefig(ldir0 + f'u10n_std_{mon}_std.png',bbox_inches='tight')
    n=n+1
    plt.show()



v10n_std = {}
namespace = globals()
v10n_std_list=[]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    v10n_std[month[m-1]] = v10n_std_months.sel(month=m)
    v10n_std[month[m-1]] = v10n_std[month[m-1]].values
    # v10n_std[month[m-1]], lon = shiftgrid(180., v10n[month[m-1]], lon, start=False)
    # sst_list.append(sst.isel(month=m).values) #this works for separating the months
    # namespace[f'sst_mean_{month[m]}'] = ds_mean[m] #separates the 12 dataarrays by name
    print('this worked')

#argh fix the scale
# sst[month[m]] = sst[month[m]].values - 273


# ssm = ssm
# mss = mss
# ticks_v10n_std_mean = [sm,m,ms]
# ticks_up=np.arange(0,31,step=5)
# gfont = {'fontsize' : 16}

n=1
for mon in months:
    lon = ds.v10n_std_months.lon.values
    lat = ds.v10n_std_months.lat.values
    v10n_std = ds.v10n_std_months.sel(month=n).values
    v10n_std, lon = shiftgrid(180., v10n_std, lon, start=False)
    mu = round(np.nanmean(v10n_std),2)
    sigma = round(np.nanstd(v10n_std),2)
    v10n_std_min = round(np.nanmin(v10n_std),2)
    v10n_std_max = round(np.nanmax(v10n_std),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10n_std_min:.2f}, max is {v10n_std_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_v10n = [0,m,ms,mss]
    # lons = ds_mean.lon.values
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lon, lat, v10n_std, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
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
    plt.title(f"ERA5 v10n 1979-2020 {mon} Standard Deviation",fontsize=18,pad=10)
    cf = plt.pcolormesh(lon,lat,v10n_std,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
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
    cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # cbar.ax.text(0.2,0,f'MIN\n{v10n_min:.2f}',ha='center',va='center')
    # cbar.ax.text(3.3,0.6,f'MAX\n{v10n_std_max:.2f}',ha='center',va='center')
    cbar.set_label('Neutral meridional wind at 10m (m/s)',fontsize=12)
    if not (os.path.exists(ldir0 + f'v10n_std_{mon}_std.png')):
        plt.savefig(ldir0 + f'v10n_std_{mon}_std.png',bbox_inches='tight')
    n=n+1
    plt.show()
    
    
    
u10_std = {}
namespace = globals()
u10_std_list=[]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    u10_std[month[m-1]] = u10_std_months.sel(month=m)
    u10_std[month[m-1]] = u10_std[month[m-1]].values
    # u10_std[month[m-1]], lon = shiftgrid(180., u10[month[m-1]], lon, start=False)
    # sst_list.append(sst.isel(month=m).values) #this works for separating the months
    # namespace[f'sst_mean_{month[m]}'] = ds_mean[m] #separates the 12 dataarrays by name
    print('this worked')

#argh fix the scale
# sst[month[m]] = sst[month[m]].values - 273


# ssm = ssm
# mss = mss
# ticks_u10_std_mean = [sm,m,ms]
# ticks_up=np.arange(0,31,step=5)
# gfont = {'fontsize' : 16}

n=1
for mon in months:
    lon = ds.u10_std_months.lon.values
    lat = ds.u10_std_months.lat.values
    u10_std = ds.u10_std_months.sel(month=n).values
    u10_std, lon = shiftgrid(180., u10_std, lon, start=False)
    mu = round(np.nanmean(u10_std),2)
    sigma = round(np.nanstd(u10_std),2)
    u10_std_min = round(np.nanmin(u10_std),2)
    u10_std_max = round(np.nanmax(u10_std),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10_std_min:.2f}, max is {u10_std_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_u10 = [0,m,ms,mss]
    # lons = ds_mean.lon.values
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lon, lat, u10_std, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
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
    plt.title(f"ERA5 u10 1979-2020 {mon} Standard Deviation",fontsize=18,pad=10)
    cf = plt.pcolormesh(lon,lat,u10_std,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
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
    cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # cbar.ax.text(0.2,0,f'MIN\n{u10_min:.2f}',ha='center',va='center')
    # cbar.ax.text(3.3,0.6,f'MAX\n{u10_std_max:.2f}',ha='center',va='center')
    cbar.set_label('Zonal wind at 10m (m/s)',fontsize=12)
    if not (os.path.exists(ldir0 + f'u10_std_{mon}_std.png')):
        plt.savefig(ldir0 + f'u10_std_{mon}_std.png',bbox_inches='tight')
    n=n+1
    plt.show()


v10_std = {}
namespace = globals()
v10_std_list=[]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    v10_std[month[m-1]] = v10_std_months.sel(month=m)
    v10_std[month[m-1]] = v10_std[month[m-1]].values
    # v10_std[month[m-1]], lon = shiftgrid(180., v10[month[m-1]], lon, start=False)
    # sst_list.append(sst.isel(month=m).values) #this works for separating the months
    # namespace[f'sst_mean_{month[m]}'] = ds_mean[m] #separates the 12 dataarrays by name
    print('this worked')

#argh fix the scale
# sst[month[m]] = sst[month[m]].values - 273


# ssm = ssm
# mss = mss
# ticks_v10_std_mean = [sm,m,ms]
# ticks_up=np.arange(0,31,step=5)
# gfont = {'fontsize' : 16}

n=1
for mon in months:
    lon = ds.v10_std_months.lon.values
    lat = ds.v10_std_months.lat.values
    v10_std = ds.v10_std_months.sel(month=n).values
    v10_std, lon = shiftgrid(180., v10_std, lon, start=False)
    mu = round(np.nanmean(v10_std),2)
    sigma = round(np.nanstd(v10_std),2)
    v10_std_min = round(np.nanmin(v10_std),2)
    v10_std_max = round(np.nanmax(v10_std),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {v10_std_min:.2f}, max is {v10_std_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_v10 = [0,m,ms,mss]
    # lons = ds_mean.lon.values
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    CS = plt.contour(lon, lat, v10_std, transform=ccrs.PlateCarree(),levels=[m,ms,mss],
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
    plt.title(f"ERA5 v10 1979-2020 {mon} Standard Deviation",fontsize=18,pad=10)
    cf = plt.pcolormesh(lon,lat,v10_std,transform=ccrs.PlateCarree(),cmap=cmocean.cm.amp,
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
    cbar2.ax.set_xticklabels(['0','$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # cbar.ax.text(0.2,0,f'MIN\n{v10_min:.2f}',ha='center',va='center')
    # cbar.ax.text(3.3,0.6,f'MAX\n{v10_std_max:.2f}',ha='center',va='center')
    cbar.set_label('Meridional wind at 10m (m/s)',fontsize=12)
    if not (os.path.exists(ldir0 + f'full_v10_std_{mon}_std.png')):
        plt.savefig(ldir0 + f'v10_std_{mon}_std.png',bbox_inches='tight')
    n=n+1
    plt.show()











#and now for ilhas std



#now for the ilhas region
from matplotlib.patches import Circle
ldir0 = '/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly/'
fl = ldir0 + 'monthly_means_stds.nc'
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
u10n_ilhas1 = ds.u10n_mean_months.sel(lat=slice(-10,10),lon=slice(300,359.75))
u10n_ilhas2 = ds.u10n_mean_months.sel(lat=slice(-10,10),lon=slice(0,15))
u10n_ilhas = xr.concat([u10n_ilhas1,u10n_ilhas2],dim='lon')

#plotting monthly means
lon = u10n_ilhas.lon.values
lat = u10n_ilhas.lat.values
lon2d, lat2d = np.meshgrid(lon, lat)


u10n = {}
namespace = globals()
sst_list=[]
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in np.arange(1,13):
    u10n[month[m-1]] = u10n_ilhas.sel(month=m)
print('this worked')


ticks_sst_mean = [sm,m,ms]
ticks_up=np.arange(0,31,step=5)
gfont = {'fontsize' : 16}



n=0
for mon in u10n.keys():
    mu = round(np.nanmean(u10n[mon]),2)
    sigma = round(np.nanstd(u10n[mon]),2)
    u10n_min = round(np.nanmin(u10n[mon]),2)
    u10n_max = round(np.nanmax(u10n[mon]),2)
    print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {u10n_min:.2f}, max is {u10n_max:.2f}')
    [ssm,sm,m,ms,mss] = [np.around(mu-2*sigma,decimals=2),
             np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2),
             np.around(mu+2*sigma,decimals=2)]
    print([ssm,sm,m,ms,mss])
    ticks_u10n_mean = [sm,m,ms]
    # lons = u10n_ilhas.lon
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
    cf = plt.pcolormesh(lon2d, lat2d, u10n_ilhas[n],transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmocean.cm.delta)
    plt.title(f"ERA5 Neutral Zonal Wind 1979-2020 {mon} mean",fontdict = gfont,pad = 20)
    gl = ax1.gridlines(draw_labels=True,xlocs=np.arange(-60, 16, 15),
                      ylocs=np.arange(-10, 11, 5),
                      linewidth=0.25,color='black',zorder = 7,alpha=0.7,linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    gl.ypadding = 10
    gl.xpadding = 10
    # Add archipelago
    patches = [Circle((-29.34583333, 0.91694444), radius=0.25, color='red')]
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
    cbar.ax.set_xticklabels([f'$\mu$-$\sigma$ = {sm:.1f}', f'$\mu$ = {m:.1f}',
                              f'$\mu$+$\sigma$ = {ms:.1f}'],fontsize=12)
    # cbar.ax.text(22.2,27,f'MIN\n{u10n_min:.2f}',ha='center',va='center')
    # cbar.ax.text(31.8,27,f'MAX\n{u10n_max:.2f}',ha='center',va='center')
    CS = ax1.contour(lon, lat, u10n[mon], transform=ccrs.PlateCarree(),levels=[ssm,sm,m,ms,mss],
                      linewidths=0.35,colors='k',zorder=1,inline=True)
    fmt = {} 
    strs = ['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$','$\mu$+$\sigma$','$\mu$+2$\sigma$']
    for l, s in zip(CS.levels, strs): 
        fmt[l] = s
    ax1.clabel(CS, CS.levels, inline=True,fontsize=6,fmt=fmt)
    if not (os.path.exists(ldir0 + f'ilhas_monthly_u10n_{mon}_mean.png')):
        plt.savefig(ldir0 + f'ilhas_monthly_u10n_{mon}_mean.png',bbox_inches='tight')
    n=n+1