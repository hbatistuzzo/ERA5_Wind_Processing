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


# ldir0 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/download.nc"
# ds = xr.open_dataset(ldir0)

# u10n = ds.u10n.sel(expver=1)
# v10n = ds.v10n.sel(expver=1)
# u10 = ds.u10.sel(expver=1)
# v10 = ds.v10.sel(expver=1)


# from dask.diagnostics import ProgressBar
# from dask.distributed import Client
# client = Client() #ok lets try this
# client

# with ProgressBar():
#     u10n_mean_months = u10n.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 

# infile = open(ldir0 + 'u10n_mean_months.pckl', 'wb')
# pickle.dump(u10n_mean_months, infile)
# infile.close()


# v10n_mean_months = v10n.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
# u10_mean_months = u10.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
# v10_mean_months = v10.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 


# u10n_std_months = u10n.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
# v10n_std_months = v10n.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
# u10_std_months = u10.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
# v10_std_months = v10.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load()

# #Create a neat dataset
# lat = ds.u10n_mean_months.lat.values
# lon = ds.u10n_mean_months.lon.values
# month = ds.u10n_mean_months.month.values
# ddd2 = {'month':{'dims': 'month','data': month, 'attrs': {'units': 'none'}},
#         'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N'}},
#        'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E'}}}

# from collections import OrderedDict as od
# z_c1 = od()
# z_c2 = od()
# z_c3 = od()
# z_c4 = od()
# z_c1['u10n_mean_months'] = ds.u10n_mean_months.values
# z_c1['v10n_mean_months'] = ds.v10n_mean_months.values
# z_c2['u10_mean_months'] = ds.u10_mean_months.values
# z_c2['v10_mean_months'] = ds.v10_mean_months.values
# z_c3['u10n_std_months'] = ds.u10n_std_months.values
# z_c3['v10n_std_months'] = ds.v10n_std_months.values
# z_c4['u10_std_months'] = ds.u10_std_months.values
# z_c4['v10_std_months'] = ds.v10_std_months.values


# encoding = {}
# for key in z_c1.keys():
#     ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c1[key],
#                           'attrs': {'units': 'm/s'}}})
#     encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
#                                 'zlib': True, '_FillValue': -9999999}})

# for key in z_c2.keys():
#     ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c2[key],
#                           'attrs': {'units': 'm/s'}}})
#     encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
#                                 'zlib': True, '_FillValue': -9999999}})
# for key in z_c3.keys():
#     ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c3[key],
#                           'attrs': {'units': 'm/s'}}})
#     encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
#                                 'zlib': True, '_FillValue': -9999999}})

# for key in z_c4.keys():
#     ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c4[key],
#                           'attrs': {'units': 'm/s'}}})
#     encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
#                                 'zlib': True, '_FillValue': -9999999}})
# ds = xr.Dataset.from_dict(ddd2)
# ds.to_netcdf('/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly_means_stds.nc', format='NETCDF4',
#              encoding=encoding)


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


# for u10
z = u10_mean.values

u10 = {}
namespace = globals()
u10_list=[]
for m in np.arange(0,12):
    u10[month[m]] = u10_mean[m,:,:].values
    u10_list.append(u10_mean[m,:,:].values) #this works for separating the months
    namespace[f'u10_mean_{month[m]}'] = u10_mean[m] #separates the 12 dataarrays by name
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
ticks_u10 = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in u10.keys():
    lons = u10_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = u10[mon]
    
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
    plt.title(f'ERA5 u10 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_u10,extend='both',pad=0.1)
    cbar.set_label('Zonal wind at 10m (m/s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_u10,extend='both',pad=0.1)
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
    if not (os.path.exists(ldir0 + f'u10_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'u10_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()


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
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = v10[mon]
    
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
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.7,0.3,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.7,0.3,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir0 + f'v10_{mon}_mean_global.png')):
        plt.savefig(ldir0 + f'v10_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()

#create movies





################################# STRESS ###################################
ldir1 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/stress.nc"
ds = xr.open_dataset(ldir1)


ewss = ds.ewss.sel(expver=1) #Eastward Turbulent Surface Stress
iews = ds.iews.sel(expver=1) #Instantaneous Eastward Turbulent Surface Stress
inss = ds.inss.sel(expver=1) #Instantaneous Northward Turbulent Surface Stress
nsss = ds.nsss.sel(expver=1) #Northward turbulent surface stress

from dask.distributed import Client
client = Client() #ok lets try this

tic()
ewss_mean_months = ewss.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load()
toc()

iews_mean_months = iews.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
inss_mean_months = inss.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 
nsss_mean_months = nsss.groupby('time.month').mean(dim='time',skipna=True,keep_attrs=False).load() 


ewss_std_months = ewss.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
iews_std_months = iews.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
inss_std_months = inss.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load() 
nsss_std_months = nsss.groupby('time.month').std(dim='time',skipna=True,keep_attrs=False).load()


#Create a neat dataset
lat = ewss_mean_months.latitude.values
lon = ewss_mean_months.longitude.values
month = ewss_mean_months.month.values
ddd2 = {'month':{'dims': 'month','data': month, 'attrs': {'units': 'none'}},
        'lat': {'dims': 'lat','data': lat, 'attrs': {'units': 'deg N'}},
        'lon': {'dims': 'lon', 'data': lon, 'attrs': {'units': 'deg E'}}}

from collections import OrderedDict as od
z_c1 = od()
z_c2 = od()
z_c3 = od()
z_c4 = od()
z_c1['ewss_mean_months'] = ewss_mean_months.values
z_c1['nsss_mean_months'] =  nsss_mean_months.values
z_c2['iews_mean_months'] = iews_mean_months.values
z_c2['inss_mean_months'] = inss_mean_months.values
z_c3['ewss_std_months'] = ewss_std_months.values
z_c3['nsss_std_months'] = nsss_std_months.values
z_c4['iews_std_months'] = iews_std_months.values
z_c4['inss_std_months'] = inss_std_months.values


encoding = {}
for key in z_c1.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c1[key],
                          'attrs': {'units': 'N m**-2 s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c2.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c2[key],
                          'attrs': {'units': 'N m**-2'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
for key in z_c3.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c3[key],
                          'attrs': {'units': 'N m**-2 s'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})

for key in z_c4.keys():
    ddd2.update({key: {'dims': ('month','lat', 'lon'), 'data': z_c4[key],
                          'attrs': {'units': 'N m**-2'}}})
    encoding.update({key: {'dtype': 'int16', 'scale_factor': 0.0001,
                                'zlib': True, '_FillValue': -9999999}})
ds = xr.Dataset.from_dict(ddd2)
ds.load().to_netcdf('/media/hbatistuzzo/DATA/Ilhas/Era5/stress/stress_monthly_means_stds.nc', format='NETCDF4',
              encoding=encoding)

# from humanize import naturalsize as ns
# from sys import getsizeof
# ds_size = humanize.naturalsize(ds.nbytes)

#check

ldir2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/stress/"
ds = xr.open_dataset(ldir2+'stress_monthly_means_stds.nc')


month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

ewss_mean = ds.ewss_mean_months
nsss_mean = ds.nsss_mean_months
iews_mean = ds.iews_mean_months
inss_mean = ds.inss_mean_months


lats = ewss_mean[0].lat.values
lons = ewss_mean[0].lon.values

# for ewss
z = ewss_mean.values

ewss = {}
namespace = globals()
ewss_list=[]
for m in np.arange(0,12):
    ewss[month[m]] = ewss_mean[m,:,:].values
    ewss_list.append(ewss_mean[m,:,:].values) #this works for separating the months
    namespace[f'ewss_mean_{month[m]}'] = ewss_mean[m] #separates the 12 dataarrays by name
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
ticks_ewss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in ewss.keys():
    lons = ewss_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = ewss[mon]
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Mollweide())
    # CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
    #                  levels=[ssm,sm,m,ms,mss],zorder=1)
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
    plt.title(f'ERA5 ewss 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_ewss,extend='both',pad=0.1)
    cbar.set_label('Eastward Turbulent Surface Stress (N m**-2 s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_ewss,extend='both',pad=0.1)
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
    # if not (os.path.exists(ldir2 + f'ewss_{mon}_mean_global.png')):
    #     plt.savefig(ldir2 + f'ewss_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()



# for nsss
z = nsss_mean.values

nsss = {}
namespace = globals()
nsss_list=[]
for m in np.arange(0,12):
    nsss[month[m]] = nsss_mean[m,:,:].values
    nsss_list.append(nsss_mean[m,:,:].values) #this works for separating the months
    namespace[f'nsss_mean_{month[m]}'] = nsss_mean[m] #separates the 12 dataarrays by name
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
ticks_nsss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in nsss.keys():
    lons = nsss_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = nsss[mon]
    
    z, lons = shiftgrid(180., z, lons, start=False)
    ax = plt.axes(projection=ccrs.Mollweide())
    # CS = plt.contour(lons, lats, z, transform=ccrs.PlateCarree(),linewidths=0.5,colors='k',
    #                  levels=[ssm,sm,m,ms,mss],zorder=1)
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
    plt.title(f'ERA5 nsss 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_nsss,extend='both',pad=0.1)
    cbar.set_label('Northward turbulent surface stress(N m**-2 s)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_nsss,extend='both',pad=0.1)
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
    if not (os.path.exists(ldir2 + f'nsss_{mon}_mean_global.png')):
        plt.savefig(ldir2 + f'nsss_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()


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
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = iews[mon]
    
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
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.05,0,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.05,0,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir2 + f'iews_{mon}_mean_global.png')):
        plt.savefig(ldir2 + f'iews_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()


# for inss
z = inss_mean.values

inss = {}
namespace = globals()
inss_list=[]
for m in np.arange(0,12):
    inss[month[m]] = inss_mean[m,:,:].values
    inss_list.append(inss_mean[m,:,:].values) #this works for separating the months
    namespace[f'inss_mean_{month[m]}'] = inss_mean[m] #separates the 12 dataarrays by name
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
ticks_inss = [ssm,sm,m,ms,mss]
gfont = {'fontname':'Helvetica','fontsize' : 16}



for mon in inss.keys():
    lons = inss_mean[0].lon.values
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = inss[mon]
    
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
    plt.title(f'ERA5 inss 1979-2020 {mon} mean',fontdict = {'fontsize' : 24},family='Verdana',pad=15);
    cf = plt.pcolormesh(lons,lats,z,transform=ccrs.PlateCarree(),cmap=cmocean.cm.delta,
                        vmin=ssm,vmax=mss,zorder=0)
    cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_inss,extend='both',pad=0.1)
    cbar.set_label('Instantaneous Northward Turbulent Surface Stress (N m**-2)')
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect('auto')
    ax2 = cbar.ax.twiny()
    cbar.ax.set_position(pos)
    ax2.set_position(pos)
    cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",ticks=ticks_inss,extend='both',pad=0.1)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.set_xticklabels(['$\mu$-2$\sigma$', '$\mu$-$\sigma$', '$\mu$',
                              '$\mu$+$\sigma$','$\mu$+2$\sigma$']) 
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # gl.ylocator = mticker.FixedLocator([-40,-30,-20,-10,0])
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ssm-0.03,0,f'MIN\n{z_min:.2f}',ha='center',va='center')
    cbar.ax.text(mss+0.03,0,f'MAX\n{z_max:.2f}',ha='center',va='center')
    ax.set_aspect('auto')
    if not (os.path.exists(ldir2 + f'inss_{mon}_mean_global.png')):
        plt.savefig(ldir2 + f'inss_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()





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

path2 = r"/media/hbatistuzzo/DATA/Ilhas/Era5/wind/monthly_means_stds.nc"
ds = xr.open_dataset(path2)

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
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Mollweide())
    z = u10n[mon]
    
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
    # if not (os.path.exists(ldir0 + f'u10n_{mon}_mean_global.png')):
    #     plt.savefig(ldir0 + f'u10n_{mon}_mean_global.png',bbox_inches='tight')
    plt.show()
plt.show()






















# u10n MEAN
lon = ds.u10n_mean_months.lon.values
lat = ds.u10n_mean_months.lat.values
u10n = ds.u10n_mean_months.values

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




########################## and doing it again

ldir0='/media/hbatistuzzo/DATA/Ilhas/CMC_GHRSST/'
sst_mean = xr.open_dataset(ldir0+'full_mean.nc')
sst_std = xr.open_dataset(ldir0+'full_std.nc')

#plotting
lon = sst_mean.lon.values
lat = sst_mean.lat.values

data_sst_mean = sst_mean.sst_mean.values
data_sst_std1 = sst_std.sst_mean.values
data_sst_std2 = sst_std.sst_std.values

#for sst_mean
lon2d, lat2d = np.meshgrid(lon, lat)
datar = data_sst_mean[:,:]

mu = np.nanmean(datar)
sigma = np.nanstd(datar)
sst_min = np.nanmin(datar)
sst_max =np.nanmax(datar)
print(f'mean is {mu:.2f}, std is {sigma:.2f}, min is {sst_min:.2f}, max is {sst_max:.2f}')


[sm,m,ms] = [np.around(mu-sigma,decimals=2),
             np.around(mu,decimals=2),
             np.around(mu+sigma,decimals=2)]
print([sm,m,ms])

#for the colorbar levels
ticks_sst_mean = [sm,m,ms]
gfont = {'fontsize' : 16}

fig = plt.figure(figsize=(8,6),dpi=300)
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(resolution='50m', color='black', linewidth=0.25)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.05,zorder=2)
ax.add_feature(cfeature.LAND, edgecolor='black',linewidths=0.05,zorder=3)
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='steelblue', edgecolor='black', linewidths=0.1,zorder=4,alpha=0.5)
ax.add_feature(cfeature.BORDERS, linewidths=0.1,zorder=5)
ax.add_feature(cfeature.RIVERS, linewidths=0.25,zorder=6)
ax.add_feature(cartopy.feature.OCEAN)
ax.set_global()
cf = plt.pcolormesh(lon2d, lat2d, datar,transform=ccrs.PlateCarree(), shading='auto',cmap=cmocean.cm.thermal)
plt.title("CMC-GHRSST 'Analysed SST' 1991-2017 mean",fontdict = gfont,pad = 20)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,xlocs=np.arange(-120, 121, 60),
                  ylocs=np.arange(-90, 91, 30),
                  linewidth=0.25,color='black',zorder = 7)
# gl.xlocator = mticker.FixedLocator([-180, -90,-45, 0, 45, 90,180])
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.rotate_labels = False
gl.ypadding = 30
gl.xpadding = 10
cbar = plt.colorbar(ax=ax,orientation="horizontal",ticks=ticks_sst_mean,extend='both')
cbar.set_label('Analysed SST (\u00B0C))')
pos = cbar.ax.get_position()
cbar.ax.set_aspect('auto')
ax2 = cbar.ax.twiny()
cbar.ax.set_position(pos)
ax2.set_position(pos)
cbar2 = plt.colorbar(cax=ax2, orientation="horizontal",extend='both')
cbar2.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels(['$\mu$-$\sigma$ = 3.31', '$\mu$ = 14.46',
                          '$\mu$+$\sigma$ = 25.62'])

cbar.ax.get_yaxis().set_ticks([])
# cbar.ax.text(sm-8.5,15,f'MIN\n{sst_min:.2f}',ha='center',va='center')
# cbar.ax.text(ms+8.5,15,f'MAX\n{sst_max:.2f}',ha='center',va='center')
# if not (os.path.exists(ldir0 + 'full_sst_mean.png')):
#     plt.savefig(ldir0 + 'full_sst_mean.png',bbox_inches='tight')
plt.show()















