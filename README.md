# ERA5_Wind_Processing

![GitHub top language](https://img.shields.io/github/languages/top/hbatistuzzo/Era5_Wind_Processing)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hbatistuzzo/Era5_Wind_Processing)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/hbatistuzzo/Era5_Wind_Processing)
![GitHub last commit](https://img.shields.io/github/last-commit/hbatistuzzo/Era5_Wind_Processing)

- Python 3.8.13
	- Numpy 1.20.3
	- Seaborn 0.11.2
	- Matplotlib 3.5.3
	- Xarray 2022.10.0
	- Dask 2021.09
	- Cartopy 0.21.0
	- Cmocean 2.0

<img src="ECMWF.png" align="right" width="50%"/>
ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1950 to present. ERA5 is produced by the Copernicus Climate Change Service (C3S) at ECMWF.

<br/>
<br/>

ERA5 provides hourly estimates of a large number of atmospheric, land and oceanic climate variables. The data cover the Earth on a 30km grid and resolve the
atmosphere using 137 levels from the surface up to a height of 80km. ERA5 includes information about uncertainties for all variables at reduced spatial and
temporal resolutions.

<br/>
In this project I've created some functions to help plot zonal and meridional wind data. The code also performs some previous statistics on the dataset to allow 
the plotting of mean +/- standard deviation contours, which are also applied in the colorbar.
<p align="center"><img src="u10_full_mean.png"alt="full"  width="75%"></p>


There is also some snippets to plot "zoomed-in" data in a regular lat-long cartesian projection, as below:

<p align = "center">
<img src="ilhas_full_u10n_mean2.png" alt="ilhas2" width="75%">
</p>
<p align = "center">
The dot marks the location of the St. Peter & St. Paul archipelago
</p>

<br/>
It uses dask to optimize the code run (as opening these many netcdf files in nested concatenation is not a trivial task). Even less trivial is performing the chronological
mean for the monthly climatology. Big kudos to those who created this awesome tool.
<br/>
Finally, it also spits out Hovmoller diagrams for each of the latitudes of interest, since they are useful for identifying meso and large-scale phenomena such as vortices and planetary (Rossby/Equatorial Kelvin) waves.
<br/>
I managed to do some neat animations with matplotlib and cartopy!
<p align="center"><img src="https://im3.ezgif.com/tmp/ezgif-3-058b56cfa6.gif" width="100%" alt="cake"></p>


