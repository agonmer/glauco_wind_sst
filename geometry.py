import numpy as np
def grad_sphere(field, llon, llat):
    """
    Function to calculate the gradient of a 2D scalar field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field = np.double(field)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)
    
    df_dx = field-field
    df_dx[:,1:-1] = (field[:,2:]-field[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    df_dx[:,0] = (field[:,1]-field[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    df_dx[:,-1] = (field[:,-1]-field[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    df_dy = field-field
    df_dy[1:-1,:] = (field[2:,:]-field[:-2,:])/(R*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    df_dy[0,:] = (field[1,:]-field[0,:])/(R*(llat[1,:]-llat[0,:])*np.pi/180)
    df_dy[-1,:] = (field[-1,:]-field[-2,:])/(R*(llat[-1,:]-llat[-2,:])*np.pi/180)
    
    return df_dx, df_dy

def div_sphere(field_a, field_b, llon, llat):
    """
    Function to calculate the divergence of a 2D vectorial field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field_a = np.double(field_a)
    field_b = np.double(field_b)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)

    div_a = field_a-field_a
    div_a[:,1:-1] = (field_a[:,2:]-field_a[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    div_a[:,0] = (field_a[:,1]-field_a[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    div_a[:,-1] = (field_a[:,-1]-field_a[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    div_b = field_b-field_b
    div_b[1:-1,:] = (field_b[2:,:]*costheta[2:,:]-field_b[:-2,:]*costheta[:-2,:])/(R*costheta[1:-1,:]*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    div_b[0,:] = (field_b[1,:]*costheta[1,:]-field_b[0,:]*costheta[0,:])/(R*costheta[0,:]*(llat[1,:]-llat[0,:])*np.pi/180)
    div_b[-1,:] = (field_b[-1,:]*costheta[-1,:]-field_b[-2,:]*costheta[-2,:])/(R*costheta[-1,:]*(llat[-1,:]-llat[-2,:])*np.pi/180)
        
    div = div_a + div_b
    return div
    
def stretched_div_sphere(field_a, field_b, llon, llat, u_field, v_field):
    """
    Function to calculate the divergence of a 2D vectorial field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The coordinates are stretched with the u_field and the v_field components.
    The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.

    field_a = np.double(field_a)
    field_b = np.double(field_b)
    llon = np.double(llon)
    llat = np.double(llat)
    u_field = np.double(u_field)
    v_field = np.double(v_field)
    
    costheta = np.cos(llat*np.pi/180)

    div_a = field_a-field_a
    div_a[:,1:-1] = (field_a[:,2:]-field_a[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    div_a[:,0] = (field_a[:,1]-field_a[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    div_a[:,-1] = (field_a[:,-1]-field_a[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    div_b = field_b-field_b
    div_b[1:-1,:] = (field_b[2:,:]*costheta[2:,:]-field_b[:-2,:]*costheta[:-2,:])/(R*costheta[1:-1,:]*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    div_b[0,:] = (field_b[1,:]*costheta[1,:]-field_b[0,:]*costheta[0,:])/(R*costheta[0,:]*(llat[1,:]-llat[0,:])*np.pi/180)
    div_b[-1,:] = (field_b[-1,:]*costheta[-1,:]-field_b[-2,:]*costheta[-2,:])/(R*costheta[-1,:]*(llat[-1,:]-llat[-2,:])*np.pi/180)
        
    div = np.abs(u_field)*div_a + np.abs(v_field)*div_b
    return div

def nan_gaussian_filter(field,sigma):
    """
    Function to smooth the field ignoring the NaNs.
    I follow the first answer here 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    By default, the filter is truncated at 4 sigmas.
    """
    from scipy.ndimage import gaussian_filter
    
    field = np.double(field)
    
    # Take the original field and replace the NaNs with zeros.
    field0 = field.copy()
    field0[np.isnan(field)] = 0
    ff = gaussian_filter(field0, sigma=sigma)
    
    # Create the smoothed weight field.
    weight = 0*field.copy()+1
    weight[np.isnan(field)] = 0
    ww = gaussian_filter(weight, sigma=sigma)
    
    zz = ff/(ww*weight) # This rescale for the actual weights used in the filter and set to NaN where the field
                        # was originally NaN.
    return zz

def L2wind_2_regular_grid_mask(lon_wind,lat_wind,u,v,lon_sst,lat_sst,extent_param):
    """
    Function to interpolate the wind from the scatterometer to the SST grid using a linear approach 
    separating the two halves of the scatterometer observation. It also discard conditions when less than 4 points
    are to be interpolated (which is impossible with the griddata function). It avoids the fake interpolated data
    generated by griddata within the convex hull of the set of points creating a mask using the original lon lat of 
    the wind data.
    """
    import numpy as np
    from scipy.interpolate import griddata, interp1d
    from scipy.ndimage import correlate

    swath_width = np.size(u,1) # The 0-th axis is the length of the swath.
    mid = int(np.round(swath_width/2))

    lon1_wind = lon_wind[:,:mid]
    lon2_wind = lon_wind[:,mid:]
    lat1_wind = lat_wind[:,:mid]
    lat2_wind = lat_wind[:,mid:]

    lon1_area = lon1_wind[(lon_wind[:,:mid]>extent_param[0]) & (lon_wind[:,:mid]<extent_param[1]) & 
                          (lat_wind[:,:mid]>extent_param[2]) & (lat_wind[:,:mid]<extent_param[3])]
    lon2_area = lon2_wind[(lon_wind[:,mid:]>extent_param[0]) & (lon_wind[:,mid:]<extent_param[1]) & 
                          (lat_wind[:,mid:]>extent_param[2]) & (lat_wind[:,mid:]<extent_param[3])]
    lat1_area = lat1_wind[(lon_wind[:,:mid]>extent_param[0]) & (lon_wind[:,:mid]<extent_param[1]) & 
                          (lat_wind[:,:mid]>extent_param[2]) & (lat_wind[:,:mid]<extent_param[3])]
    lat2_area = lat2_wind[(lon_wind[:,mid:]>extent_param[0]) & (lon_wind[:,mid:]<extent_param[1]) & 
                          (lat_wind[:,mid:]>extent_param[2]) & (lat_wind[:,mid:]<extent_param[3])]

    u1 = u[:,:mid]
    u2 = u[:,mid:]

    u1_area = u1[(lon_wind[:,:mid]>extent_param[0]) & (lon_wind[:,:mid]<extent_param[1]) & 
                 (lat_wind[:,:mid]>extent_param[2]) & (lat_wind[:,:mid]<extent_param[3])]
    u2_area = u2[(lon_wind[:,mid:]>extent_param[0]) & (lon_wind[:,mid:]<extent_param[1]) & 
                 (lat_wind[:,mid:]>extent_param[2]) & (lat_wind[:,mid:]<extent_param[3])]

    llon, llat = np.meshgrid(lon_sst,lat_sst)

    if (np.isnan(u1_area).all()) and (np.isnan(u2_area).all()):
        u_interp = np.nan
    elif (len(u1_area)<4) and (len(u2_area)<4): # In this case no interpolation can be performed.
        u_interp = np.nan
    else:
        if (np.isnan(u2_area).all()) or (len(u2_area)<4):
            u_interp = griddata((lon1_area,lat1_area),u1_area,(llon,llat))
            # Remove the fake points within the convex hull using a mask.
            f_upper = interp1d(lon1_wind[:,-1],lat1_wind[:,-1],kind='linear',fill_value='extrapolate')
            f_lower = interp1d(lon1_wind[:,0],lat1_wind[:,0],kind='linear',fill_value='extrapolate')
            valid_mask = (llat<f_upper(lon_sst)) & (llat>f_lower(lon_sst))
            u_interp[~valid_mask] = np.nan
        elif (np.isnan(u1_area).all()) or (len(u1_area)<4):
            u_interp = griddata((lon2_area,lat2_area),u2_area,(llon,llat))
            # Remove the fake points within the convex hull.
            f_upper = interp1d(lon2_wind[:,-1],lat2_wind[:,-1],kind='linear',fill_value='extrapolate')
            f_lower = interp1d(lon2_wind[:,0],lat2_wind[:,0],kind='linear',fill_value='extrapolate')
            valid_mask = (llat<f_upper(lon_sst)) & (llat>f_lower(lon_sst))
            u_interp[~valid_mask] = np.nan
        else:
            u1_interp = griddata((lon1_area,lat1_area),u1_area,(llon,llat))
            # Remove the fake points within the convex hull.
            f1_upper = interp1d(lon1_wind[:,-1],lat1_wind[:,-1],kind='linear',fill_value='extrapolate')
            f1_lower = interp1d(lon1_wind[:,0],lat1_wind[:,0],kind='linear',fill_value='extrapolate')
            valid_mask1 = (llat<f1_upper(lon_sst)) & (llat>f1_lower(lon_sst))
            u1_interp[~valid_mask1] = np.nan

            u2_interp = griddata((lon2_area,lat2_area),u2_area,(llon,llat))
            # Remove the fake points within the convex hull.
            f2_upper = interp1d(lon2_wind[:,-1],lat2_wind[:,-1],kind='linear',fill_value='extrapolate')
            f2_lower = interp1d(lon2_wind[:,0],lat2_wind[:,0],kind='linear',fill_value='extrapolate')
            valid_mask2 = (llat<f2_upper(lon_sst)) & (llat>f2_lower(lon_sst))
            u2_interp[~valid_mask2] = np.nan

            u_interp = u1_interp; np.putmask(u_interp,~np.isnan(u2_interp),u2_interp)            
        # At the edge of the swath there might be some discontinuities: remove them by setting those
        # points to NaN (2 points are removed). We do this with a squared filter on the NaN mask of the 
        # original swath.
        if ~np.isnan(u_interp).all():
            first_mask_nan = u_interp/u_interp
            footprint = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            mask_nan = correlate(first_mask_nan,footprint/25)
            u_interp = u_interp*mask_nan
        # Alternatively, we can try to remove one grid point from the side of the mid when we select the u1 and u2
        # subfields. But this is not enough.  Even if I remove 1 point from both sides, some edge effect remains.

        
    u1 = v[:,:mid]
    u2 = v[:,mid:]

    u1_area = u1[(lon_wind[:,:mid]>extent_param[0]) & (lon_wind[:,:mid]<extent_param[1]) & 
                 (lat_wind[:,:mid]>extent_param[2]) & (lat_wind[:,:mid]<extent_param[3])]
    u2_area = u2[(lon_wind[:,mid:]>extent_param[0]) & (lon_wind[:,mid:]<extent_param[1]) & 
                 (lat_wind[:,mid:]>extent_param[2]) & (lat_wind[:,mid:]<extent_param[3])]

    if (np.isnan(u1_area).all()) and (np.isnan(u2_area).all()):
        v_interp = np.nan
    elif (len(u1_area)<4) and (len(u2_area)<4): # In this case no interpolation can be performed.
        v_interp = np.nan
    else:
        if (np.isnan(u2_area).all()) or (len(u2_area)<4):
            v_interp = griddata((lon1_area,lat1_area),u1_area,(llon,llat))
            v_interp[~valid_mask] = np.nan
        elif (np.isnan(u1_area).all()) or (len(u1_area)<4):
            v_interp = griddata((lon2_area,lat2_area),u2_area,(llon,llat))
            v_interp[~valid_mask] = np.nan
        else:
            u1_interp = griddata((lon1_area,lat1_area),u1_area,(llon,llat))
            u1_interp[~valid_mask1] = np.nan

            u2_interp = griddata((lon2_area,lat2_area),u2_area,(llon,llat))
            u2_interp[~valid_mask2] = np.nan

            v_interp = u1_interp; np.putmask(v_interp,~np.isnan(u2_interp),u2_interp)            
        # At the edge of the swath there might be some discontinuities: remove them by setting those
        # points to NaN (2 points are removed). We do this with a squared filter on the NaN mask of the 
        # original swath.
        if ~np.isnan(v_interp).all():
            first_mask_nan = v_interp/v_interp
            footprint = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            mask_nan = correlate(first_mask_nan,footprint/25)
            v_interp = v_interp*mask_nan
        # Alternatively, we can try to remove one grid point from the side of the mid when we select the u1 and u2
        # subfields. But this is not enough.  Even if I remove 1 point from both sides, some edge effect remains.
            
    return u_interp, v_interp

def crop_field(field,lon,lat,area):
    """
    Function to crop a 2d field over the area specified by the list
    area = [minlon,maxlon,minlat,maxlat]
    """
    valid_indices = np.argwhere((lon>area[0])&(lon<area[1])&(lat>area[2])&(lat<area[3]))
    x_min, y_min = valid_indices.min(axis=0)
    x_max, y_max = valid_indices.max(axis=0)
    return field[x_min:x_max+1, y_min:y_max+1]
