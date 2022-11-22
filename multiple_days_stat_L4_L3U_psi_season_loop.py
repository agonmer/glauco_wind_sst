# This is to compute systematically the statistics for the DMM over different
# areas of the globe and over single seasons. It can also be used for the PA.
# The standard deviation of the gaussian filter used to determine the
# background wind is sigma. The background wind is saved in the output files and
# both L3U and L4 data are considered, as raw data and filtered with a gaussian
# kernel of standard deviation psi.

# Set the geographical parameters for the analysis and the figures.
# Look at the areas of the WBCs defined by O'Neill et al, J. Cli (2012).
# The western Pacific area is defined following Li and Carbone, JAS (2012).
# The eastern Pacific cold tongue are follows Chelton et al., J. Cli (2001).

#                                                                 # DMM           # PA            # u',sst'
#area_str = 'tropical_atlantic'; area_name = 'Tropical Atlantic'
#minlon = -62.; maxlon = 15.; minlat = -20.; maxlat = 20.; 
#area_str = 'extended_eureca'; area_name = 'extended EUREC4A'     # DONE          # DONE
#minlon = -62.; maxlon = -40.; minlat = 0; maxlat = 20.; 
#area_str = 'eurec4a'; area_name = 'EUREC4A'
#minlon = -62.; maxlon = -48.; minlat = 4.; maxlat = 16.;
area_str = 'gulf_stream'; area_name = 'Gulf Stream'              # DONE          # DONE           # DONE  
minlon = -83.; maxlon = -30.; minlat = 30.; maxlat = 55.;
area_str = 'malvinas'; area_name = 'Malvinas current'            # DONE          # DONE
minlon = -70.; maxlon = 0.; minlat = -60.; maxlat = -30.;
#area_str = 'agulhas'; area_name = 'Agulhas current'              # DONE          # DONE
#minlon = 0.; maxlon = 100.; minlat = -60.; maxlat = -30.;
#area_str = 'kuroshio'; area_name = 'Kuroshio current'            # DONE          # DONE
#minlon = 140.; maxlon = 180.; minlat = 30.; maxlat = 50.;
#area_str = 'western_pacific'; area_name = 'western Pacific'
#minlon = 130.; maxlon = 160.; minlat = 0.; maxlat = 15.;
#area_str = 'eastern_pacific'; area_name = 'eastern Pacific'      # DONE
#minlon = -160.; maxlon = -75.; minlat = -10.; maxlat = 10.;
#area_str = 'med'; area_name = 'Mediterranean Sea'                # DONE          # DONE 
#minlon = -5.6; maxlon = 43.4; minlat = 30; maxlat=47.5;

# Select here the fields to be analysed.
str_a = 'sst_prime' # Choose between: 'gamma', 'dsst_dr', 'lapl_sst', 'd2sst_ds2', 'sst_prime'
str_b = 'ws_prime' # Choose between: 'wind_div', 'dr_dot_prime_dr', 'ds_dot_prime_ds', 'ws_prime'

# Select the standard deviation of the Gaussian filter used to determine the background wind field.
sigma = 10 # Take a relatively local sigma, to highlight the small scale features. If we take 5 the correlation
# seems to be less significant: we problably go to too fine scales... Standard value: sigma = 10.
# In the tropics the SST structures are large: check how the results change for different sigmas.
sigma = 25 # for the Gulf Stream u',sst' - we keep it for all WBCs for u',sst'

# Set the loop on time, separating the seasons for multipe years.
# L2 ASCAT winds available between 1st Jan 2007 to end of March 2014.
list_str_start = ['2007-03-02','2007-06-01','2007-09-01',
                  '2007-12-01','2008-03-01','2008-06-01','2008-09-01',
                  '2008-12-01','2009-03-01','2009-06-01','2009-09-01',
                  '2009-12-01','2010-03-01','2010-06-01','2010-09-01',
                  '2010-12-01','2011-03-01','2011-06-01','2011-09-01',
                  '2011-12-01','2012-03-01','2012-06-01','2012-09-01',
                  '2012-12-01','2013-03-01','2013-06-01','2013-09-01',
                  '2013-12-01']
list_str_end = ['2007-05-31','2007-08-31','2007-11-30',
                '2008-02-28','2008-05-31','2008-08-31','2008-11-30',
                '2009-02-28','2009-05-31','2009-08-31','2009-11-30',
                '2010-02-28','2010-05-31','2010-08-31','2010-11-30',
                '2011-02-28','2011-05-31','2011-08-31','2011-11-30',
                '2012-02-28','2012-05-31','2012-08-31','2012-11-30',
                '2013-02-28','2013-05-31','2013-08-31','2013-11-30',
                '2014-02-28']

##### THAT'S IT #####

# Import the relevant modules.
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import xarray as xr
import pandas as pd
import glob
import datetime as dt
import gzip
import cdsapi
import tarfile
import h5netcdf
from numpy import random
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
from scipy import stats
from scipy.ndimage import correlate
import geometry as gm # grad_sphere, div_sphere, nan_gaussian_filter, L2wind_2_regular_grid_mask
import distributions as dstr # mixed_distribution, mixed_distribution_with_hist (also with two samples)
from matplotlib.offsetbox import AnchoredText

# Set some relevant paths.
path2CCI_CDR = '/media/agostino/sailboat/'
path2SST = path2CCI_CDR + 'neodc/esacci/sst/data/CDR_v2/'
path2wind = path2CCI_CDR + 'podaac/OceanWinds/'
path2ascat = path2wind + 'ascat/L2/metop_a/cdr/12km/'
path2output = path2CCI_CDR + 'glauco/output_txt/'

# Set some parameters for the maps.
extent_param = [minlon, maxlon, minlat, maxlat]
crs = ccrs.PlateCarree()

def plot_background(ax):
    ax.set_extent(extent_param)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    #ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 14}
    gl.ylabel_style = {'fontsize': 14}
    return ax

if area_str=='western_pacific':
    fig_height = np.max([4,np.abs(maxlat-minlat)/2.])
    fig_width = np.abs(maxlon-minlon)/2.
else:
    fig_height = np.max([4,np.abs(maxlat-minlat)/5.])
    fig_width = np.abs(maxlon-minlon)/5.

def read_and_interpolate_the_wind(filename_wind,lon_sst,lat_sst):
    """
    Read the content from a L2 ASCAT wind file and interpolate it on the (llon,llat) grid.
    """
    hfile = gzip.open(filename_wind, "rb")
    ds_wind = xr.open_dataset(hfile)

    # Set the longitude in the range between -180 and 180.
    ds_wind_newlon = ds_wind.assign_coords(lon=(((ds_wind.lon + 180) % 360) - 180))

    # Read the variables of interest.
    lon_wind = ds_wind_newlon.lon.values
    lat_wind = ds_wind_newlon.lat.values
    wind_speed = ds_wind_newlon.wind_speed.values
    wind_dir = ds_wind_newlon.wind_dir.values
    u = wind_speed * np.cos(np.pi*0.5-wind_dir*np.pi/180)
    v = wind_speed * np.sin(np.pi*0.5-wind_dir*np.pi/180)

    # Remove the wind data with any high quality flag.
    wvc_quality_flag = ds_wind_newlon.wvc_quality_flag.values
    u[np.log2(wvc_quality_flag)>5]=np.nan
    v[np.log2(wvc_quality_flag)>5]=np.nan

    # Interpolate the wind obs over the regular SST grid.
    u_interp, v_interp = gm.L2wind_2_regular_grid_mask(lon_wind,lat_wind,u,v,lon_sst,lat_sst,extent_param)

    return u_interp, v_interp

def compute_two_fields(str_a,str_b,sigma,llon,llat,l3_sst,u_interp,v_interp):
    """
    Compute the fields defined by str_a and str_b using the l3_sst, u_interp and v_interp variables,
    which are all defined on the same grid (llon,llat).
    Note that the fields are treated as lists, because the append function
    is much faster with respect to the numpy.append function. The lists are converted to numpy arrays at the end.
    """
    # Get the background wind field.
    smooth_u = gm.nan_gaussian_filter(u_interp,sigma)
    smooth_v = gm.nan_gaussian_filter(v_interp,sigma)
    smooth_ws = np.sqrt(smooth_u**2+smooth_v**2)

    cosphi = smooth_u/smooth_ws
    sinphi = smooth_v/smooth_ws

    # Get the anomalies with respect to the background wind field.
    u_prime = u_interp-smooth_u
    v_prime = v_interp-smooth_v

    dsst_dx, dsst_dy = gm.grad_sphere(l3_sst,llon,llat)
    if str_a=='gamma':
        a_prime = u_interp*dsst_dx + v_interp*dsst_dy
    elif str_a=='dsst_dr':
        a_prime = dsst_dx*cosphi + dsst_dy*sinphi
    elif str_a=='lapl_sst':
        a_prime = gm.div_sphere(dsst_dx,dsst_dy,llon,llat)
    elif str_a=='d2sst_ds2':
        dsst_ds = -dsst_dx*sinphi + dsst_dy*cosphi
        ddsst_ds_dx, ddsst_ds_dy = gm.grad_sphere(dsst_ds,llon,llat)
        a_prime = -ddsst_ds_dx*sinphi + ddsst_ds_dy*cosphi
    elif str_a=='sst_prime':
        smooth_sst = gm.nan_gaussian_filter(l3_sst,sigma)
        a_prime = l3_sst-smooth_sst
		
    if str_b=='wind_div':
        b_prime = gm.div_sphere(u_interp,v_interp,llon,llat)
    elif str_b=='dr_dot_prime_dr':
        r_dot_prime = u_prime*cosphi + v_prime*sinphi
        dr_dot_prime_dx, dr_dot_prime_dy = gm.grad_sphere(r_dot_prime,llon,llat)
        b_prime = dr_dot_prime_dx*cosphi + dr_dot_prime_dy*sinphi 
    elif str_b=='ds_dot_prime_ds':
        s_dot_prime = -u_prime*sinphi + v_prime*cosphi
        ds_dot_prime_dx, ds_dot_prime_dy = gm.grad_sphere(s_dot_prime,llon,llat)
        b_prime = -ds_dot_prime_dx*sinphi + ds_dot_prime_dy*cosphi
    elif str_b=='ws_prime':
        b_prime = np.sqrt(u_interp**2+v_interp**2)-smooth_ws


    # Remove the NaNs, from the variables to be concatenated (with no subsampling).
    #a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
    #b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
    #U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    

    return a_prime, b_prime, smooth_ws #a_to_be_concat, b_to_be_concat, U_to_be_concat

##################################################
# L4 raw and smoothed analysis.
##################################################

print('----- L4 analyses -----')

for tt,str_start in enumerate(list_str_start):
    str_end = list_str_end[tt]

    # psi is the standard deviation of the Gaussian filter to be applied to the SST field.
    # Use sigma to determine the background wind field.
    for psi in range(0,4,2):
        print('==============================')
        print('psi = '+str(psi))

        if psi==0:
            file2save = area_str + '_allvalues_' + str_b + '_vs_' + str_a + '_from_' + str_start + '_to_' + str_end + '_sigma' + str(sigma) + '.txt'
        else:
            file2save = area_str + '_allvalues_' + str_b + '_vs_' + str_a + '_from_' + str_start + '_to_' + str_end + '_sigma' + str(sigma) + '_psi' + str(psi) + '.txt'

        # Check whether the file with the variables of interest, in the region and time period of interest exists.
        # If the file is there, do not compute it, if it does not, compute and save it.

        # Check whether the file exists.
        if os.path.exists('/media/agostino/twiga-polimi-3/glauco/output_txt/'+file2save):
            # Read the existing file.
            print('The file '+file2save+' exists')
        else: # Compute it.
            a = []; # Forcing field.
            b = []; # Atmospheric response field.
            U = []; # Background wind speed.

            instant_start = np.datetime64(str_start)
            instant_end = np.datetime64(str_end)

            instant = instant_start

            while instant<=instant_end:
                print(instant)
                print('---------------------------------------------------------------------------------------')

                # Read the SST file.
                pd_instant = pd.to_datetime(instant)
                year_oi = str(pd_instant.year).zfill(4)
                month_oi = str(pd_instant.month).zfill(2)
                day_oi = str(pd_instant.day).zfill(2)
                doy = dt.datetime(int(year_oi), int(month_oi), int(day_oi)).timetuple().tm_yday
                doy_str = str(doy).zfill(3) # 3 digits string.

                # Set the path to the SST data. For a given day, read the L4 daily map over the area of interest
                path_sst_oi = path2SST + '/Analysis/L4/v2.1/' +  year_oi + '/' + month_oi + '/' + day_oi + '/'
                filename_sst = sorted(glob.glob(path_sst_oi + '*2.1-v*')) # this removes the file with the 'anomaly' in their name.

                ds_sst = xr.open_dataset(filename_sst[0])
                lon_sst = ds_sst.lon.sel(lon=slice(extent_param[0],extent_param[1])).values
                lat_sst = ds_sst.lat.sel(lat=slice(extent_param[2],extent_param[3])).values
                l4_sst_orig = ds_sst.analysed_sst[0].sel(lon=slice(extent_param[0],extent_param[1])).sel(lat=slice(extent_param[2],extent_param[3])).values
                if psi==0:
                    l4_sst = l4_sst_orig
                else:
                    l4_sst = gm.nan_gaussian_filter(l4_sst_orig,psi)
                llon, llat = np.meshgrid(lon_sst,lat_sst)

                # Read all the wind swaths over the same area during the day considered.
                # Set the path to the wind data.
                path_wind_oi = path2ascat + '/' + year_oi + '/' + doy_str + '/'
                filenames_wind = sorted(glob.glob(path_wind_oi + '*nc.gz'))

                for name in filenames_wind:
                    hfile = gzip.open(name, "rb")
                    ds_wind = xr.open_dataset(hfile) 

                    # Set the longitude in the range between -180 and 180.
                    ds_wind_newlon = ds_wind.assign_coords(lon=(((ds_wind.lon + 180) % 360) - 180))

                    # Read the variables of interest.
                    lon_wind = ds_wind_newlon.lon.values
                    lat_wind = ds_wind_newlon.lat.values
                    wind_speed = ds_wind_newlon.wind_speed.values
                    wind_dir = ds_wind_newlon.wind_dir.values
                    u = wind_speed * np.cos(np.pi*0.5-wind_dir*np.pi/180)
                    v = wind_speed * np.sin(np.pi*0.5-wind_dir*np.pi/180)

                    # Remove the wind data with any high quality flag.
                    wvc_quality_flag = ds_wind_newlon.wvc_quality_flag.values
                    u[np.log2(wvc_quality_flag)>5]=np.nan
                    v[np.log2(wvc_quality_flag)>5]=np.nan

                    # Interpolate the wind obs over the regular SST grid.
                    u_interp, v_interp = gm.L2wind_2_regular_grid_mask(lon_wind,lat_wind,u,v,lon_sst,lat_sst,extent_param)

                    if (~np.isnan(u_interp).all()) and (~np.isnan(v_interp).all()):

                        # Get the background wind field.
                        smooth_u = gm.nan_gaussian_filter(u_interp,sigma)
                        smooth_v = gm.nan_gaussian_filter(v_interp,sigma)
                        smooth_ws = np.sqrt(smooth_u**2+smooth_v**2)

                        cosphi = smooth_u/smooth_ws
                        sinphi = smooth_v/smooth_ws

                        # Get the anomalies with respect to the background wind field.
                        u_prime = u_interp-smooth_u
                        v_prime = v_interp-smooth_v

                        dsst_dx, dsst_dy = gm.grad_sphere(l4_sst,llon,llat)
                        if str_a=='gamma':
                            a_prime = u_interp*dsst_dx + v_interp*dsst_dy
                            x_string = 'u.grad(SST) [K/s]'; vmin_a=-2.2e-4; vmax_a=2.2e-4
                        elif str_a=='dsst_dr':
                            a_prime = dsst_dx*cosphi + dsst_dy*sinphi
                            x_string = 'dSST/dr [K/m]'; vmin_a=-2.2e-5; vmax_a=2.2e-5
                        elif str_a=='lapl_sst':
                            a_prime = gm.div_sphere(dsst_dx,dsst_dy,llon,llat)
                            x_string = 'lapl SST [K/m^2]'; vmin_a=-1e-9; vmax_a=1e-9
                        elif str_a=='d2sst_ds2':
                            dsst_ds = -dsst_dx*sinphi + dsst_dy*cosphi
                            ddsst_ds_dx, ddsst_ds_dy = gm.grad_sphere(dsst_ds,llon,llat)
                            a_prime = -ddsst_ds_dx*sinphi + ddsst_ds_dy*cosphi
                            x_string = 'd2SST/ds2 [K/m^2]'; vmin_a=-1e-9; vmax_a=1e-9
                        elif str_a=='sst_prime':
                            smooth_sst = gm.nan_gaussian_filter(l4_sst,sigma)
                            a_prime = l4_sst-smooth_sst
                            x_string = 'SST_prime [K]'; vmin_a=-5; vmax_a=5

                        if str_b=='wind_div':
                            b_prime = gm.div_sphere(u_interp,v_interp,llon,llat)
                            y_string = 'Wind divergence [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
                        elif str_b=='dr_dot_prime_dr':
                            r_dot_prime = u_prime*cosphi + v_prime*sinphi
                            dr_dot_prime_dx, dr_dot_prime_dy = gm.grad_sphere(r_dot_prime,llon,llat)
                            b_prime = dr_dot_prime_dx*cosphi + dr_dot_prime_dy*sinphi 
                            y_string = 'dr dot prime/dr [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
                        elif str_b=='ds_dot_prime_ds':
                            s_dot_prime = -u_prime*sinphi + v_prime*cosphi
                            ds_dot_prime_dx, ds_dot_prime_dy = gm.grad_sphere(s_dot_prime,llon,llat)
                            b_prime = -ds_dot_prime_dx*sinphi + ds_dot_prime_dy*cosphi
                            y_string = 'ds dot prime/ds [1/s]'; vmin_b=-2.2e-4; vmax_b=2.2e-4
                        elif str_b=='ws_prime':
                            b_prime = np.sqrt(u_interp**2+v_interp**2)-smooth_ws
                            y_string = 'ws_prime [m/s]'; vmin_b=-5; vmax_b=5;

                        controlname = str_a
                        varname = str_b

                        # Concatenate the variables (with no subsampling) removing the NaNs.
                        a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
                        b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
                        U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    

                        a.extend(a_to_be_concat)
                        b.extend(b_to_be_concat)
                        U.extend(U_to_be_concat)

                    print(name)

                print('---------------------------------------------------------------------------------------')
                instant += np.timedelta64(1,'D')

            a = np.array(a)
            b = np.array(b)
            U = np.array(U)

            # Save the a and b variables as text files.
            a[np.isinf(a)] = np.nan
            b[np.isinf(b)] = np.nan
            U[np.isinf(U)] = np.nan

            # Remove the NaN to make the files smaller, otherwise the final files are unreadable.
            a, b, U = a[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], b[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], U[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))] 

            d = {'control':np.transpose(a), 'response':np.transpose(b), 'background_ws':np.transpose(U)}
            df = pd.DataFrame(data=d)
            df.to_csv('/media/agostino/twiga-polimi-3/glauco/output_txt/'+file2save, index=False)
            # If in the name nothing is specified about the SST, the L4 analysis data are used.

##################################################
# L3U raw and smoothed analysis.
##################################################

print('----- L3U analyses -----')

for tt,str_start in enumerate(list_str_start):
    str_end = list_str_end[tt]

    # psi is the standard deviation of the Gaussian filter to be applied to the SST field.
    # Use sigma to determine the background wind field.
    for psi in range(0,4,2):
        print('==============================')
        print('psi = '+str(psi))

        if psi==0:
            file2save = area_str + '_allvalues_' + str_b + '_vs_' + str_a + '_from_' + str_start + '_to_' + str_end + '_sigma' + str(sigma) + '_L3Usst.txt'
        else:
            file2save = area_str + '_allvalues_' + str_b + '_vs_' + str_a + '_from_' + str_start + '_to_' + str_end + '_sigma' + str(sigma) + '_L3Usst_psi' + str(psi) + '.txt'

        # Check whether the file with the variables of interest, in the region and time period of interest exists.
        # If the file is there, do not compute it, if it does not, compute and save it.

        # Check whether the file exists.
        if os.path.exists('/media/agostino/twiga-polimi-3/glauco/output_txt/'+file2save):
            # Read the existing file.
            print('The file '+file2save+' exists')
        else: # Compute it.
            a = []; # Forcing field.
            b = []; # Atmospheric response field.
            U = []; # Background wind speed.

            instant_start = np.datetime64(str_start)
            instant_end = np.datetime64(str_end)

            instant = instant_start

            # Initialize some variables to compute the time average of the a_prime and b_prime fields.
            Ny = int((extent_param[3]-extent_param[2])/0.05)
            Nx = int((extent_param[1]-extent_param[0])/0.05)
            sum_field_a = np.zeros([Ny,Nx])
            sum_field_b = np.zeros([Ny,Nx])
            time_count_field_a = np.zeros([Ny,Nx])
            time_count_field_b = np.zeros([Ny,Nx])

            while instant<=instant_end:
                print(instant)
                print('---------------------------------------------------------------------------------------')

                # Set the time handle.
                pd_instant = pd.to_datetime(instant)
                year_oi = str(pd_instant.year).zfill(4)
                month_oi = str(pd_instant.month).zfill(2)
                day_oi = str(pd_instant.day).zfill(2)
                doy = dt.datetime(int(year_oi), int(month_oi), int(day_oi)).timetuple().tm_yday
                doy_str = str(doy).zfill(3) # 3 digits string.

                # Read all the wind files available during the day considered.
                # Make an array of the reference time of each wind file, so that it is faster to match the time while looping
                # on the SST files.
                path_wind_oi = path2ascat + '/' + year_oi + '/' + doy_str + '/'
                filenames_wind = sorted(glob.glob(path_wind_oi + '*nc.gz'))
                time_wind = []
                for name_wind in filenames_wind:
                    instant_wind = np.datetime64(name_wind[-51:-47]+'-'+name_wind[-47:-45]+'-'+name_wind[-45:-43]+'T'+
                                                 name_wind[-42:-40]+':'+name_wind[-40:-38]+':'+name_wind[-38:-36])
                    time_wind.append(instant_wind)
                #print(time_wind)

                # Set the path to the SST data. For a given day, list the files.
                # Try to match the wind files of the same day, looking at their reference time and selecting those that exist
                # within a  one-hour interval, choosing their relative time order according to the orbit type (ascending or
                # descending).
                # Then, if less than 1 match has been found, check whether the reference time is before 1AM. In that case,
                # look for the last wind file of the previous day, as there might be an overlap. Similarly, if it is after 23 
                # PM, look for the first wind file of the following day, for the same reason.
                path_sst_oi = path2SST + '/AVHRR/L3U/v2.1/AVHRRMTA_G/' +  year_oi + '/' + month_oi + '/' + day_oi + '/'
                filenames_sst = sorted(glob.glob(path_sst_oi + '*nc'))    

                for name_sst in filenames_sst:
                    match = 0 # Count the matches between the SST and wind swaths: we need to get one!
                    instant_sst = np.datetime64(name_sst[-74:-70]+'-'+name_sst[-70:-68]+'-'+name_sst[-68:-66]+'T'+
                                                name_sst[-66:-64]+':'+name_sst[-64:-62]+':'+name_sst[-62:-60])
                    print('SST file:' + str(instant_sst))

                    # Read the coordinates and the SST field of the SST file.
                    ds_sst = xr.open_dataset(name_sst)
                    ds_sst.close()
                    lon_sst = ds_sst.lon.sel(lon=slice(extent_param[0],extent_param[1])).values
                    lat_sst = ds_sst.lat.sel(lat=slice(extent_param[2],extent_param[3])).values
                    llon, llat = np.meshgrid(lon_sst,lat_sst)
                    ql_area = ds_sst.quality_level.sel(lon=slice(extent_param[0],extent_param[1])).sel(lat=slice(extent_param[2], extent_param[3]))
                    # Remove the data with QL lower than 3 and filter the SST field.
                    l3u_sst_orig = ds_sst.sea_surface_temperature.where(ql_area>=3).values[0]
                    if psi==0:
                        l3u_sst = l3u_sst_orig
                    else:
                        l3u_sst = gm.nan_gaussian_filter(l3u_sst_orig,psi)

                    # Continue if there are at least ten valid points in the SST map, according to the area and the 
                    # quality flag.
                    #if ~np.isnan(l3u_sst).all():
                    if (np.nansum(~np.isnan(l3u_sst))>10)&(len(filenames_wind)>0):

                        # Try to determine whether the orbit is ascending or descending based on the dtime values.
                        # We cannot do it on the lat values because they are regular in L3U data.            
                        sst_deltatime = ds_sst.sst_dtime[0]/np.timedelta64(1,'h')
                        sst_dtime = sst_deltatime.sel(lon=slice(extent_param[0],extent_param[1]),lat=slice(extent_param[2],extent_param[3])).values
                        lat_deriv_time = sst_dtime[1:,:]-sst_dtime[:-1,:]
                        time_increment = np.nanmean(lat_deriv_time)            
                        if np.isnan(time_increment):   
                            #raise ValueError("I cannot determine whether the orbit is ascending or descending")
                            print("I cannot determine whether the orbit is ascending or descending")
                            continue
                        else:
                            if np.sign(time_increment)>0:
                                ascending_orbit=True
                                print('Ascending orbit')
                            else:
                                ascending_orbit=False
                                print('Descending orbit')        

                        # Get the time difference in minutes between the SST reference time and all the wind reference times.
                        # We expect an overlap to be present if the files are less than an hour apart.
                        # Select the sign of the time difference according to the orbit type: if the orbit is ascending, the 
                        # SST file precedes the wind file; if the orbit is descending, the SST file follows the wind file.
                        time_diff = (instant_sst - time_wind)/np.timedelta64(1, 'm')
                        if ascending_orbit:
                            files_within_an_hour = (time_diff<0) & (time_diff>-60)
                        else:
                            files_within_an_hour = (time_diff>0) & (time_diff<60)

                        for x in (jj for jj in range(len(filenames_wind)) if files_within_an_hour[jj]):
                            name_wind = filenames_wind[x]
                            instant_wind = np.datetime64(name_wind[-51:-47]+'-'+name_wind[-47:-45]+'-'+name_wind[-45:-43]+'T'+
                                                         name_wind[-42:-40]+':'+name_wind[-40:-38]+':'+name_wind[-38:-36])
                            print('SST file:' + str(instant_sst) + ' to be matched with the wind file at '+str(instant_wind))

                            # Read the wind components and interpolate them on the SST grid.
                            u_interp, v_interp = read_and_interpolate_the_wind(name_wind,lon_sst,lat_sst)

                            # If they are not empty, compute the appropriate derivatives to do the stat.
                            if (~np.isnan(u_interp).all()) and (~np.isnan(v_interp).all()):
									
                                a_prime, b_prime, smooth_ws = compute_two_fields(str_a,str_b,sigma,llon,llat,l3u_sst,u_interp,v_interp)   
                                a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
                                b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
                                U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
                                
                                sum_field_a = np.nansum([sum_field_a,a_prime],axis=0)
                                sum_field_b = np.nansum([sum_field_b,b_prime],axis=0)
                                time_count_field_a = np.nansum([time_count_field_a,a_prime/a_prime],axis=0)
                                time_count_field_b = np.nansum([time_count_field_b,b_prime/b_prime],axis=0)

                                a.extend(a_to_be_concat)
                                b.extend(b_to_be_concat)
                                U.extend(U_to_be_concat)

                            match += 1

                        # If you have not found any match yet, look for it in the previous or in the following day.
                        # If a match is found here it has to be correct.
                        if match<1:
                            print('Try to see in the previous or following day')
                            minutes_after_midnight = (instant_sst-instant)/np.timedelta64(1,'m')
                            minutes_before_next_midnight = (instant+np.timedelta64(1,'D')-instant_sst)/np.timedelta64(1,'m')

                            if minutes_after_midnight<60:

                                # Get the previous day.
                                pd_previous_day = pd.to_datetime(instant - np.timedelta64(1,'D'))
                                year_pvd = str(pd_previous_day.year).zfill(4)
                                month_pvd = str(pd_previous_day.month).zfill(2)
                                day_pvd = str(pd_previous_day.day).zfill(2)
                                doy_pvd = dt.datetime(int(year_pvd), int(month_pvd), int(day_pvd)).timetuple().tm_yday
                                doy_str_pvd = str(doy_pvd).zfill(3) # 3 digits string.
                                print('doy_str_pvd:'+doy_str_pvd)
                                # List the wind files of the previous day.
                                path_wind_pvd = path2ascat + '/' + year_pvd + '/' + doy_str_pvd + '/'
                                filenames_wind_pvd = sorted(glob.glob(path_wind_pvd + '*nc.gz'))
                                if len(filenames_wind_pvd)==0:
                                    print('No data available for the previous day')
                                    continue

                                # Read the time information of the last wind file of the previous day.
                                # If the files are closer than one hour try to match the fields.
                                name_wind = filenames_wind_pvd[-1] # Last file of the previous day.
                                instant_wind = np.datetime64(name_wind[-51:-47]+'-'+name_wind[-47:-45]+'-'+
                                                             name_wind[-45:-43]+'T'+name_wind[-42:-40]+':'+
                                                             name_wind[-40:-38]+':'+name_wind[-38:-36])
                                if (instant_sst-instant_wind)/np.timedelta64(1,'m')<60:
                                    print('TRY TO MATCH WITH WIND FILE AT '+str(instant_wind))
                                    # Read the wind components and interpolate them on the SST grid.
                                    u_interp, v_interp = read_and_interpolate_the_wind(name_wind,lon_sst,lat_sst)

                                    # If they are not empty, compute the appropriate derivatives to do the stat.
                                    if (~np.isnan(u_interp).all()) and (~np.isnan(v_interp).all()):

                                        a_prime, b_prime, smooth_ws = compute_two_fields(str_a,str_b,sigma,llon,llat,l3u_sst,u_interp,v_interp)                    
                                        a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
                                        b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
                                        U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
                                        sum_field_a = np.nansum([sum_field_a,a_prime],axis=0)
                                        sum_field_b = np.nansum([sum_field_b,b_prime],axis=0)
                                        time_count_field_a = np.nansum([time_count_field_a,a_prime/a_prime],axis=0)
                                        time_count_field_b = np.nansum([time_count_field_b,b_prime/b_prime],axis=0)

                                        a.extend(a_to_be_concat)
                                        b.extend(b_to_be_concat)
                                        U.extend(U_to_be_concat)


                            elif minutes_before_next_midnight<60:

                                # Get the next day.
                                pd_next_day = pd.to_datetime(instant + np.timedelta64(1,'D'))
                                year_nxd = str(pd_next_day.year).zfill(4)
                                month_nxd = str(pd_next_day.month).zfill(2)
                                day_nxd = str(pd_next_day.day).zfill(2)
                                doy_nxd = dt.datetime(int(year_nxd), int(month_nxd), int(day_nxd)).timetuple().tm_yday
                                doy_str_nxd = str(doy_nxd).zfill(3) # 3 digits string.

                                # List the wind files of the next day.
                                path_wind_nxd = path2ascat + '/' + year_nxd + '/' + doy_str_nxd + '/'
                                filenames_wind_nxd = sorted(glob.glob(path_wind_nxd + '*nc.gz'))
                                if len(filenames_wind_nxd)==0:
                                    print('No data available for the next day')
                                    continue
                                # Read the time information of the first wind file of the next day.
                                # If the files are closer than one hour try to match the fields.
                                name_wind = filenames_wind_nxd[0] # First file of the next day.
                                instant_wind = np.datetime64(name_wind[-51:-47]+'-'+name_wind[-47:-45]+'-'+
                                                             name_wind[-45:-43]+'T'+name_wind[-42:-40]+':'+
                                                             name_wind[-40:-38]+':'+name_wind[-38:-36])
                                if (instant_wind-instant_sst)/np.timedelta64(1,'m')<60:
                                    print('TRY TO MATCH WITH WIND FILE AT '+str(instant_wind))
                                    # Read the wind components and interpolate them on the SST grid.
                                    u_interp, v_interp = read_and_interpolate_the_wind(name_wind,lon_sst,lat_sst)

                                    # If they are not empty, compute the appropriate derivatives to do the stat.
                                    if (~np.isnan(u_interp).all()) and (~np.isnan(v_interp).all()):

                                        a_prime, b_prime, smooth_ws = compute_two_fields(str_a,str_b,sigma,llon,llat,l3u_sst,u_interp,v_interp)                    
                                        a_to_be_concat = a_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]
                                        b_to_be_concat = b_prime[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    
                                        U_to_be_concat = smooth_ws[(~np.isnan(a_prime))&(~np.isnan(b_prime))&(~np.isnan(smooth_ws))]    

                                        sum_field_a = np.nansum([sum_field_a,a_prime],axis=0)
                                        sum_field_b = np.nansum([sum_field_b,b_prime],axis=0)
                                        time_count_field_a = np.nansum([time_count_field_a,a_prime/a_prime],axis=0)
                                        time_count_field_b = np.nansum([time_count_field_b,b_prime/b_prime],axis=0)

                                        a.extend(a_to_be_concat)
                                        b.extend(b_to_be_concat)
                                        U.extend(U_to_be_concat)



                print('---------------------------------------------------------------------------------------')
                instant += np.timedelta64(1,'D')

                # With np.append(a,a_to_be_concat)
                # One month (April 2009) over the Med takes 4min 31s, one year (2019) takes 51min 5s.
                # One season (Fall 2010) over the Gulf Stream takes 16min 54s.
                # One season (winter 2009-2012) over the Agulhas takes 37min 14s.
                # With a.extend(a_to_be_concat)
                # One season (spring 2010) over the Med takes 12min 41s.

            a = np.array(a)
            b = np.array(b)
            U = np.array(U)

            # Save the a and b variables as text files.
            a[np.isinf(a)] = np.nan
            b[np.isinf(b)] = np.nan
            U[np.isinf(U)] = np.nan

            # Remove the NaN to make the files smaller, otherwise the final files are unreadable.
            #a, b = a[np.logical_and(~np.isnan(a),~np.isnan(b))], b[np.logical_and(~np.isnan(a),~np.isnan(b))] 
            a, b, U = a[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], b[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))], U[(~np.isnan(a))&(~np.isnan(b))&(~np.isnan(U))] 

            d = {'control':np.transpose(a), 'response':np.transpose(b), 'background_ws':np.transpose(U)}
            df = pd.DataFrame(data=d)
            df.to_csv('/media/agostino/twiga-polimi-3/glauco/output_txt/'+file2save, index=False)
            # If in the name nothing is specified about the SST, the L4 analysis data are used.
