import numpy as np

def metrics_percentile_distribution(control, variable, perc_step=5):
    """
    Function to compute the slope of the linear regression based on the percentile distribution of the input variables.
    It also computes the intercept, the spearman r and its p-value.
    At this stage the slope does not consider the standard error of the mean value for each percentile.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    mean = np.nanmean(variable)
    mean_control = np.nanmean(control)
    std = np.nanstd(variable)
    std_control = np.nanstd(control)

    distribution = np.zeros(int(100/perc_step))
    std_distribution = np.zeros(int(100/perc_step))
    std_err_distribution = np.zeros(int(100/perc_step))
    distribution_control = np.zeros(int(100/perc_step))
    number_of_points = np.zeros(int(100/perc_step))
    percentiles = np.zeros(int(100/perc_step)+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
    percentiles[-1] = upper

    # Calculate the spearman r.
    sr_distr, trash = stats.spearmanr(distribution_control,distribution)
    df = len(distribution_control)-2 # degrees of freedom.
    t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
    p_value = 1 - stats.t.cdf(t_value,df=df)

    # Perform the linear regression.
    lsq_res = stats.linregress(distribution_control, distribution)
    slope = lsq_res[0]
    intercept = lsq_res[1]

    return slope, intercept, sr_distr, p_value, distribution_control, distribution

def metrics_fixed_bin_distribution(control, variable, perc_step=5):
    """
    Function to compute the slope of the linear regression based on the fixed bin distribution of the input variables.
    It also computes the intercept, the spearman r and its p-value.
    At this stage the slope does not consider the standard error of the mean value for each percentile.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    mean = np.nanmean(variable)
    mean_control = np.nanmean(control)
    std = np.nanstd(variable)
    std_control = np.nanstd(control)

    nbins = int(100/perc_step)
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width.
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

    # Calculate the spearman r.
    sr_distr, trash = stats.spearmanr(distribution_control_fb,distribution_fb)
    df = len(distribution_control_fb)-2 # degrees of freedom.
    t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
    p_value = 1 - stats.t.cdf(t_value,df=df)

    # Perform the linear regression.
    lsq_res = stats.linregress(distribution_control_fb, distribution_fb)
    slope = lsq_res[0]
    intercept = lsq_res[1]

    return slope, intercept, sr_distr, p_value, distribution_control_fb, distribution_fb

def mixed_distribution(control, variable, controlname, varname, xlimiti=np.nan, ylimiti=np.nan, str_control='', 
                       str_variable='', perc_step=5, title='', str_name=''):
    """
    Function to plot a fixed-bin distribution showing only data with a minimum number of points per bin, together
    with the corresponding percentile distribution.
    The least square linear fit is only computed for the data based on the percentile distribution.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt  

    mean = np.nanmean(variable)
    mean_control = np.nanmean(control)
    std = np.nanstd(variable)
    std_control = np.nanstd(control)
    
    nbins = int(100/perc_step) # Number of bins.
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width.
    theshold_n = 100
    
##### Fixed bin distribution: fb
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

    # Calculate the spearman r
    sr_distr_fb, trash = stats.spearmanr(distribution_control_fb,distribution_fb)
    df_fb = len(distribution_control_fb)-2 # degrees of freedom.
    t_value_fb = np.abs(sr_distr_fb)*np.sqrt((df_fb)/(1-sr_distr_fb**2))
    p_value_fb = 1 - stats.t.cdf(t_value_fb,df=df_fb)
   
    print('------------------------------------------------------')
    print('FIXED-BIN DISTRIBUTION')
    print('spearman r of the distr = ' + str(sr_distr_fb))
    print('p value spearman r of the distr = ' + str(p_value_fb))

##### Percentile distribution
    distribution = np.zeros(int(100/perc_step))
    std_distribution = np.zeros(int(100/perc_step))
    std_err_distribution = np.zeros(int(100/perc_step))
    distribution_control = np.zeros(int(100/perc_step))
    number_of_points = np.zeros(int(100/perc_step))
    percentiles = np.zeros(int(100/perc_step)+1)
#    zero_not_found = 1

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
#        if zero_not_found:
#            print(str(pp)+' percentile')
#            if cond_mean_control>0:
#                qq0 = qq # Index of the first positive cond_mean_control
#                zero_not_found = 0
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
    percentiles[-1] = upper

    perc = np.arange(perc_step/2,100,perc_step)
    
    percentiles_center = 0.5*(percentiles[:-1]+percentiles[1:])
    sr_distr, trash = stats.spearmanr(distribution_control,distribution)
    df = len(distribution_control)-2 # degrees of freedom.
    t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
    p_value = 1 - stats.t.cdf(t_value,df=df)
    
##### Figure
    
    #print(number_of_points)
    fig = plt.figure(figsize=(6., 7.))
    
    plt.errorbar(distribution_control_fb[number_of_points_fb>theshold_n],distribution_fb[number_of_points_fb>theshold_n],std_err_distribution_fb[number_of_points_fb>theshold_n],fmt='o')
    plt.errorbar(distribution_control,distribution,std_err_distribution,fmt='.',color='firebrick')

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel(str_variable, fontsize=14) #+ ' anomaly', fontsize=14)
    plt.xlabel(str_control, fontsize=14)
    #plt.ylim([-2.7,1.7])
    plt.grid()
    
    if np.isnan(xlimiti):
        xlimiti = plt.xlim()
    if np.isnan(ylimiti):
        ylimiti = plt.ylim()
    #print('mean = '+str(mean))
    plt.hlines(mean,xlimiti[0],xlimiti[1],'k',linestyles='--')
#    plt.vlines(0.5*(distribution_control[qq0-1]+distribution_control[qq0]),ylimiti[0],ylimiti[1],'k')
    plt.vlines(0,ylimiti[0],ylimiti[1],'k')

    # Linear regression on the percentile bin distribution.
    lsq_res = stats.linregress(distribution_control, distribution)
    #plt.plot(distribution_control, lsq_res[1] + lsq_res[0] * distribution_control, 'g-')
    ics = bin_centers_fb[number_of_points_fb>theshold_n]
    plt.plot(ics, lsq_res[1] + lsq_res[0] * ics, 'g-')

    plt.xlim(xlimiti)
    plt.ylim(ylimiti)
    
    print('------------------------------------------------------')
    print('PERCENTILE DISTRIBUTION')
    print('spearman r of the distr = ' + str(sr_distr))
    print('p value spearman r of the distr = ' + str(p_value))
    
    ff2 = "{:.2e}".format
    plt.annotate('y=' + str(ff2(lsq_res[1])) + ' + ' + str(ff2(lsq_res[0])) + '*x', xy=(0.1, 0.05), xycoords='axes fraction', fontsize=14)

    ff3 = "{:.2e}".format
    str_title = title + '\n FB sr_distr = '+ str(ff3(sr_distr_fb)) +' with p_val = ' + str(ff3(p_value_fb)) + '\n PB sr_distr = '+ str(ff3(sr_distr)) +' with p_val = ' + str(ff3(p_value))
    plt.title(str_title, fontsize=16)
    
    figure_name = 'mixed_distr_' + varname + '_vs_' + controlname + '_perc_step_' + str(perc_step) + '_' + str_name + '.png'
    plt.savefig('./figures/' + figure_name,bbox_inches='tight') 

    return sr_distr, p_value

def mixed_distribution_with_hist(control, variable, controlname, varname, xlimiti=np.nan, ylimiti=np.nan, str_control='', 
                                 str_variable='', perc_step=5, title='', str_name=''):
    """
    Function to plot a fixed-bin distribution showing only data with a minimum number of points per bin, together
    with the corresponding percentile distribution. The histograms of the two variables are also added.
    The least square linear fit is only computed for the data based on the percentile distribution.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    mean = np.nanmean(variable)
    mean_control = np.nanmean(control)
    std = np.nanstd(variable)
    std_control = np.nanstd(control)
    
    nbins = int(100/perc_step) # Number of bins.
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width.
    theshold_n = 100
    
##### Fixed bin distribution: fb
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

    # Calculate the spearman r
    sr_distr_fb, trash = stats.spearmanr(distribution_control_fb,distribution_fb)
    df_fb = len(distribution_control_fb)-2 # degrees of freedom.
    t_value_fb = np.abs(sr_distr_fb)*np.sqrt((df_fb)/(1-sr_distr_fb**2))
    p_value_fb = 1 - stats.t.cdf(t_value_fb,df=df_fb)
   
    print('------------------------------------------------------')
    print('FIXED-BIN DISTRIBUTION')
    print('spearman r of the distr = ' + str(sr_distr_fb))
    print('p value spearman r of the distr = ' + str(p_value_fb))

##### Percentile distribution
    distribution = np.zeros(int(100/perc_step))
    std_distribution = np.zeros(int(100/perc_step))
    std_err_distribution = np.zeros(int(100/perc_step))
    distribution_control = np.zeros(int(100/perc_step))
    number_of_points = np.zeros(int(100/perc_step))
    percentiles = np.zeros(int(100/perc_step)+1)
#    zero_not_found = 1

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
#        if zero_not_found:
#            print(str(pp)+' percentile')
#            if cond_mean_control>0:
#                qq0 = qq # Index of the first positive cond_mean_control
#                zero_not_found = 0
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
    percentiles[-1] = upper

    perc = np.arange(perc_step/2,100,perc_step)
    
    percentiles_center = 0.5*(percentiles[:-1]+percentiles[1:])
    sr_distr, trash = stats.spearmanr(distribution_control,distribution)
    df = len(distribution_control)-2 # degrees of freedom.
    t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
    p_value = 1 - stats.t.cdf(t_value,df=df)
    
##### Figure
    
    #print(number_of_points)
    
    fig = plt.figure(figsize=(10., 10.), constrained_layout=True)
    spec = fig.add_gridspec(4, 4)

    ax0 = fig.add_subplot(spec[:1, :3]) # Histogram of a.
    ax0.hist(control, bins=100)
    ax0.set_yscale('log')

    ax1 = fig.add_subplot(spec[1:, :3]) # 2d distributions.
    
    ax1.errorbar(distribution_control_fb[number_of_points_fb>theshold_n],distribution_fb[number_of_points_fb>theshold_n],std_err_distribution_fb[number_of_points_fb>theshold_n],fmt='o')
    ax1.errorbar(distribution_control,distribution,std_err_distribution,fmt='.',color='firebrick')

    #ax1.yticks(fontsize=14)
    #ax1.xticks(fontsize=14)
    ax1.set_ylabel(str_variable, fontsize=14) #+ ' anomaly', fontsize=14)
    ax1.set_xlabel(str_control, fontsize=14)
    #plt.ylim([-2.7,1.7])
    
    if np.isnan(xlimiti):
        xlimiti = ax1.get_xlim()
    if np.isnan(ylimiti):
        ylimiti = ax1.get_ylim()
    #print('mean = '+str(mean))
    ax1.hlines(mean,xlimiti[0],xlimiti[1],'k',linestyles='--')
    ax1.vlines(0,ylimiti[0],ylimiti[1],'k')

    # Linear regression on the percentile bin distribution.
    lsq_res = stats.linregress(distribution_control, distribution)
    #plt.plot(distribution_control, lsq_res[1] + lsq_res[0] * distribution_control, 'g-')
    ics = bin_centers_fb[number_of_points_fb>theshold_n]
    ax1.plot(ics, lsq_res[1] + lsq_res[0] * ics, 'g-')
    
    print('------------------------------------------------------')
    print('PERCENTILE DISTRIBUTION')
    print('spearman r of the distr = ' + str(sr_distr))
    print('p value spearman r of the distr = ' + str(p_value))
    
    ff2 = "{:.2e}".format
    plt.annotate('y=' + str(ff2(lsq_res[1])) + ' + ' + str(ff2(lsq_res[0])) + '*x', xy=(0.1, 0.05), xycoords='axes fraction', fontsize=14)

    ax2 = fig.add_subplot(spec[1:, 3:]) # Histogram of b.
    ax2.hist(variable, bins=100, orientation="horizontal")
    ax2.set_xscale('log')
    
    ylimiti0 = ax0.get_ylim()
    # Limits of the (b) panel distribution.
    ax0.vlines(xlimiti[0],ylimiti0[0],ylimiti0[1],'k')
    ax0.vlines(xlimiti[1],ylimiti0[0],ylimiti0[1],'k')
    # Mean and standard deviation of the control variable.
    ax0.vlines(mean_control,ylimiti0[0],ylimiti0[1],'r')
    ax0.vlines(mean_control-std_control,ylimiti0[0],ylimiti0[1],'r',linestyles='--')
    ax0.vlines(mean_control+std_control,ylimiti0[0],ylimiti0[1],'r',linestyles='--')

    xlimiti2 = ax2.get_xlim()
    # Limits of the (b) panel distribution.
    ax2.hlines(ylimiti[0],xlimiti2[0],xlimiti2[1],'k')
    ax2.hlines(ylimiti[1],xlimiti2[0],xlimiti2[1],'k')
    # Mean and standard deviation of the response variable.
    ax2.hlines(mean,xlimiti2[0],xlimiti2[1],'r')
    ax2.hlines(mean+std,xlimiti2[0],xlimiti2[1],'r',linestyles='--')
    ax2.hlines(mean-std,xlimiti2[0],xlimiti2[1],'r',linestyles='--')

    ax0.grid()
    ax1.grid()
    ax2.grid()
    
#    ax0.set_xlim(xlimiti)
    ax1.set_xlim(xlimiti)
    ax1.set_ylim(ylimiti)
#    ax2.set_ylim(ylimiti)

    ax0.tick_params(axis='both', labelsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)


    ff3 = "{:.2e}".format
    str_title = title + '\n FB sr_distr = '+ str(ff3(sr_distr_fb)) +' with p_val = ' + str(ff3(p_value_fb)) + '\n PB sr_distr = '+ str(ff3(sr_distr)) +' with p_val = ' + str(ff3(p_value))
    ax0.set_title(str_title, fontsize=16)

    anchored_text0 = AnchoredText('(a)', loc=2, prop=dict(size=14), frameon=False); ax0.add_artist(anchored_text0)
    anchored_text1 = AnchoredText('(b)', loc=2, prop=dict(size=14), frameon=False); ax1.add_artist(anchored_text1)
    anchored_text2 = AnchoredText('(c)', loc=2, prop=dict(size=14), frameon=False); ax2.add_artist(anchored_text2)

    
    figure_name = 'mixed_distr_with_hist_' + varname + '_vs_' + controlname + '_perc_step_' + str(perc_step) + '_' + str_name + '.png'
    plt.savefig('./figures/'+figure_name,bbox_inches='tight') 

    return sr_distr, p_value

def mixed_distribution_with_hist_two_samples(control1, variable1, control2, variable2, controlname, varname, 
                                             str_legend, xlimiti=np.nan, ylimiti=np.nan, str_control='', 
                                             str_variable='', perc_step=5, title='', str_name=''):
    """
    Function to plot a fixed-bin distribution showing only data with a minimum number of points per bin, together
    with the corresponding percentile distribution. The histograms of the two variables are also added.
    The least square linear fit is only computed for the data based on the percentile distribution.
    Two samples are considered.
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    fig = plt.figure(figsize=(10., 10.), constrained_layout=True)
    spec = fig.add_gridspec(4, 4)

    for sample in range(2):
        if sample==0:
            variable = variable1
            control = control1
            c = 'C0'
        else:
            variable = variable2
            control = control2
            c = 'C1'
     
        mean = np.nanmean(variable)
        mean_control = np.nanmean(control)
        std = np.nanstd(variable)
        std_control = np.nanstd(control)

        nbins = int(100/perc_step) # Number of bins.
        bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width.
        theshold_n = 100
    
    ##### Fixed bin distribution: fb
        distribution_fb = np.zeros(nbins)
        std_distribution_fb = np.zeros(nbins)
        std_err_distribution_fb = np.zeros(nbins)
        distribution_control_fb = np.zeros(nbins)
        number_of_points_fb = np.zeros(nbins)
        bin_edges_fb = np.zeros(nbins+1)

        for pp in range(0,100,perc_step):
            qq = int(pp/perc_step) # Index starting from 0.
            lower = np.min(control[~np.isnan(control)])+qq*bw
            upper = lower + bw
            bin_edges_fb[qq] = lower
            distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
            std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
            distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
            number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
            std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

        bin_edges_fb[-1] = upper
        bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])

        # Calculate the spearman r
        sr_distr_fb, trash = stats.spearmanr(distribution_control_fb,distribution_fb)
        df_fb = len(distribution_control_fb)-2 # degrees of freedom.
        t_value_fb = np.abs(sr_distr_fb)*np.sqrt((df_fb)/(1-sr_distr_fb**2))
        p_value_fb = 1 - stats.t.cdf(t_value_fb,df=df_fb)

    ##### Percentile distribution
        distribution = np.zeros(int(100/perc_step))
        std_distribution = np.zeros(int(100/perc_step))
        std_err_distribution = np.zeros(int(100/perc_step))
        distribution_control = np.zeros(int(100/perc_step))
        number_of_points = np.zeros(int(100/perc_step))
        percentiles = np.zeros(int(100/perc_step)+1)
    #    zero_not_found = 1

        for pp in range(0,100,perc_step):
            qq = int(pp/perc_step)
            lower = np.percentile(control[~np.isnan(control)],pp)
            upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
            percentiles[qq] = lower
            cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
            cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
            cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
    #        if zero_not_found:
    #            print(str(pp)+' percentile')
    #            if cond_mean_control>0:
    #                qq0 = qq # Index of the first positive cond_mean_control
    #                zero_not_found = 0
            distribution[qq] = cond_mean#-mean
            std_distribution[qq] = cond_std
            distribution_control[qq] = cond_mean_control#-mean_control
            number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
            std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
        percentiles[-1] = upper

        perc = np.arange(perc_step/2,100,perc_step)

        percentiles_center = 0.5*(percentiles[:-1]+percentiles[1:])
        sr_distr, trash = stats.spearmanr(distribution_control,distribution)
        df = len(distribution_control)-2 # degrees of freedom.
        t_value = np.abs(sr_distr)*np.sqrt((df)/(1-sr_distr**2))
        p_value = 1 - stats.t.cdf(t_value,df=df)

    ##### Figure
    
        if sample==0:
            ax0 = fig.add_subplot(spec[:1, :3]) # Histogram of a.
        ax0.hist(control, bins=100, density=True, alpha=0.5, color=c)
        ax0.set_yscale('log')

        if sample==0:
            ax1 = fig.add_subplot(spec[1:, :3]) # 2d distributions.
    
        ax1.errorbar(distribution_control_fb[number_of_points_fb>theshold_n],
                     distribution_fb[number_of_points_fb>theshold_n],
                     std_err_distribution_fb[number_of_points_fb>theshold_n],c=c,fmt='o')

        #ax1.yticks(fontsize=14)
        #ax1.xticks(fontsize=14)
        ax1.set_ylabel(str_variable, fontsize=14) #+ ' anomaly', fontsize=14)
        ax1.set_xlabel(str_control, fontsize=14)
        #plt.ylim([-2.7,1.7])

        if sample==0:
            if np.isnan(xlimiti).any():
                xlimiti = ax1.get_xlim()
            if np.isnan(ylimiti).any():
                ylimiti = ax1.get_ylim()
        else: # Check whether the second sample has different xlim or ylim.
            xlimiti_one = xlimiti
            ylimiti_one = ylimiti
            xlimiti_bis = ax1.get_xlim()
            ylimiti_bis = ax1.get_ylim()
            xlimiti = [np.minimum(xlimiti_one[0],xlimiti_bis[0]), np.maximum(xlimiti_one[1],xlimiti_bis[1])]
            ylimiti = [np.minimum(ylimiti_one[0],ylimiti_bis[0]), np.maximum(ylimiti_one[1],ylimiti_bis[1])]
            
        #print('mean = '+str(mean))
        ax1.hlines(mean,xlimiti[0],xlimiti[1],c,linestyles='--')
        ax1.vlines(0,ylimiti[0],ylimiti[1],c)

        # Linear regression on the percentile bin distribution.
        lsq_res = stats.linregress(distribution_control, distribution)
        #plt.plot(distribution_control, lsq_res[1] + lsq_res[0] * distribution_control, 'g-')
        ics = bin_centers_fb[number_of_points_fb>theshold_n]
        ax1.plot(ics, lsq_res[1] + lsq_res[0] * ics, '-', color=c)
        
        ff2 = "{:.2e}".format
        ax1.annotate(str_legend[sample]+ ': y=' + str(ff2(lsq_res[1])) + ' + ' + str(ff2(lsq_res[0])) + '*x', 
                     xy=(0.1, 0.05 + 0.05*sample), xycoords='axes fraction', fontsize=14)

        if sample==0:
            ax2 = fig.add_subplot(spec[1:, 3:]) # Histogram of b.
        ax2.hist(variable, bins=100, density=True, orientation="horizontal", alpha=0.5, color=c)
        ax2.set_xscale('log')
    
        ax0.grid()
        ax1.grid()
        ax2.grid()
    
        if sample==1:
            ylimiti0 = ax0.get_ylim()
            # Limits of the (b) panel distribution.
            ax0.vlines(xlimiti[0],ylimiti0[0],ylimiti0[1],'k')
            ax0.vlines(xlimiti[1],ylimiti0[0],ylimiti0[1],'k')

            xlimiti2 = ax2.get_xlim()
            # Limits of the (b) panel distribution.
            ax2.hlines(ylimiti[0],xlimiti2[0],xlimiti2[1],'k')
            ax2.hlines(ylimiti[1],xlimiti2[0],xlimiti2[1],'k')
            
        #    ax0.set_xlim(xlimiti)
            ax1.set_xlim(xlimiti)
            ax1.set_ylim(ylimiti)
        #    ax2.set_ylim(ylimiti)


        ax0.tick_params(axis='both', labelsize=14)
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)
        ax1.legend(str_legend, loc='lower right')

        str_title = title
        ax0.set_title(str_title, fontsize=16)

        anchored_text0 = AnchoredText('(a)', loc=2, prop=dict(size=14), frameon=False); 
        ax0.add_artist(anchored_text0)
        anchored_text1 = AnchoredText('(b)', loc=2, prop=dict(size=14), frameon=False); 
        ax1.add_artist(anchored_text1)
        anchored_text2 = AnchoredText('(c)', loc=2, prop=dict(size=14), frameon=False); 
        ax2.add_artist(anchored_text2)

    
        figure_name = 'mixed_distr_with_hist_' + varname + '_vs_' + controlname + '_perc_step_' + str(perc_step) + '_' + str_name + '_two_samples' + str_legend[0] +'_'+ str_legend[1]+'.png'
        plt.savefig('./figures/'+figure_name,bbox_inches='tight') 

    return sr_distr, p_value
