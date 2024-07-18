import numpy as np
import pandas as pd
import os
import re
import io
from scipy import interpolate
import matplotlib.pyplot as plt



def mulassis_spectrum_reader(filename:str):    
    '''read GCR spectrum files from SPENVIS
    Args:
        filename (str): file path

    Returns:
        pandas dataframe with Energy, Integrated Flux, Differential Flux
    '''
    file = open(filename, 'r')
    text = file.read()
    file_object = io.StringIO(text)
    cols = ['Energy_MeV_n', 'IFlux_m2_sr_s', 'DFlux_m2_sr_s_MeV_n']
    df = pd.read_csv(file_object, skiprows=71, skipfooter=1, engine='python', names=cols)
    return df




def mulassis_reader(filename:str, number_barriers:int, which_block:tuple):
    '''read GCR histogram fluence files from MULASSIS. MULASSIS outputs
    histogram information in blocks for each layer and each angle range
    (0-90, 90-180, 0-180). This function will select the desired block
    and return the information (lower energy, upper energy, mean energy,
    fluence/flux, and fluence error) as a pandas dataframe. Will also
    convert energy from keV to MeV.

    Args:
        filename: (str) path of file
        number_barriers: (int) number of barriers between layers
        which_block: (tuple) (layer number, angle number) which block you want
                    fluence information for, starting at index 0

    Returns: 
        pandas dataframe with Elow, Eup, Emean (all in MeV), Fluence/flux, flux_error
    '''
    cols = ['Elow','Eup','Emean','Fluence/flux','flux_error']
    
    file = open(filename, 'r')
    text = file.read()
    runs = text.split('End of Block')

    block_collection = {}
    number_angles = 3
    count = 0
    for lay in range(number_barriers):
        layer_string = f'layer:{lay}'
        for ang in range(number_angles):
            angle_string = f'angle:{ang}'

            selected = runs[count]
            file_object = io.StringIO(selected)
            if count == 0:
                df = pd.read_csv(file_object, skiprows=14, skipfooter=1, engine='python', names=cols)
            else:
                df = pd.read_csv(file_object, skiprows=15, skipfooter=1, engine='python', names=cols)
            
            df[['Elow', 'Eup', 'Emean']] = df[['Elow', 'Eup', 'Emean']]/1000 # keV to MeV
            block_collection[layer_string+'_'+angle_string] = df

            count+=1
    
    chosen_block = block_collection[f'layer:{which_block[0]}_angle:{which_block[1]}']

    return chosen_block




def gras_reader(filename:str, number_barriers:int, which_boundary:int, fluence_counter:bool=False, last_block:bool=False):
    '''read GCR histogram fluence files from GRAS. GRAS outputs two data blocks
    for each layer. The first just gives the total number of particles/cm2.
    The second provides the fluence (/cm2), error, and number of entries for each upper,
    lower, and mean energy bin (measured in MeV). The last data block gives information on
    electron, positron, and gamma events. This function will return the chosen block, and
    include the fluence counter block and last block if desired.
    
    Args:
        filename: (str) path of file
        number_barriers: (int) number of barriers between layers
        which_boundary: (int) layer number desired, starting at index 0
        fluence_counter: (bool) if True, returns the total number of particles/cm2
                        as part of the dataframe. Defaults to False.
        last_block: (bool) if True, returns the last block (electron, positron, gamma event
                    information) as part of the dataframe. Defaults to False.
    
    Returns:
        pandas dataframe with Elow, Eup, Emean (all in MeV), Fluence/flux (/cm2), flux_error
    '''
    cols = ['Elow','Eup','Emean','Fluence/flux','flux_error','entries']
    
    file = open(filename, 'r')
    text = file.read()
    runs = text.split('End of Block')

    block_collection = {}

    boundary_count = 0
    for lay in range((number_barriers*2)+1):
        layer_string = f'boundary:{boundary_count}'

        selected = runs[lay]
        file_object = io.StringIO(selected)
        if (lay % 2) == 0:
            if lay == (number_barriers*2):
                layer_string = 'last_block'
                df = pd.read_csv(file_object, skiprows=13, skipfooter=1, engine='python', names=['number_events', 'number_gamma', 'error_gamma',
                                                                                                 'number_electron', 'error_electron', 
                                                                                                 'number_positron', 'error_positron', 
                                                                                                 'number_steps', 'error_steps'])
            elif lay == 0:
                layer_string = f'boundary:{boundary_count}_fluence_counter'
                df = pd.read_csv(file_object, skiprows=7, skipfooter=1, engine='python', names=['scalar_fluence', 'flux_error'])
            else:
                layer_string = f'boundary:{boundary_count}_fluence_counter'
                df = pd.read_csv(file_object, skiprows=8, skipfooter=1, engine='python', names=['scalar_fluence', 'flux_error'])
        else:
            df = pd.read_csv(file_object, skiprows=34, skipfooter=1, engine='python', names=cols)
            boundary_count+=1
        block_collection[layer_string] = df


    if last_block == True:
        chosen_block = block_collection['last_block']
    
    elif fluence_counter == True:
        chosen_block = block_collection[f'boundary:{which_boundary}_fluence_counter']

    else:
        chosen_block = block_collection[f'boundary:{which_boundary}']



    return chosen_block




def normalization_factor(spectrum_df:pd.DataFrame, IFlux_array:np.array=None):
    '''find the initial particle fluence for a 1 year mission (normalization factor)
    
    Args:
        spectrum_df (DataFrame): MULASSIS dataframe returned from mulassis_reader fcn
        IFlux_array (array): if None, will calculate total initial particle fluence
                    if the IFlux DataFrame series, will calculate initial particle
                    fluence per energy value
    
    Returns: 
        initial particle fluence (float or arr) in particles/cm2
    '''
    normfactor_angular = 2.5e-1 ## angular normalization factor, from mulassis log file

    if IFlux_array is not None:
        normfactor_spectrum = IFlux_array * 3.154e7 * 4*np.pi / (100**2)  # particles/cm2/bin
    else:
        # Max Integrated flux = particles/(m2 s sr)
        # particles/(m2 s sr) * secs in 1 yr * sr in sphere / cm2 in m2 --> particles/cm2
        normfactor_spectrum = spectrum_df['IFlux_m2_sr_s'].max() * 3.154e7 * 2*np.pi / (100**2)
    return normfactor_spectrum * normfactor_angular




def frac_attenuated_per_thickness(data_path:str, thickness_search_string:str, which_system:str, 
                                  number_barriers:int, before_after_layers:tuple, threshold_val:float=0, gas_weight:float=None):
    """Determine the fraction of initial GCR fluence that is attenuated after passing through a layer of material
    with a certain thickness. Reads in data files for all thickness (or column density) values within one folder.

    Args:
        data_path (str): path of folder with all MULASSIS returns for a specific material
        thickness_search_string (str): string used to search for the thickness/column density values, given in the name of each file
        which_system (str): ('mulassis' or 'gras') either MULASSIS or GRAS type of files
        number_barriers (int): number of barriers (including vacuum) the data contains
        before_after_layers (tuple): which layers to choose the before and after values from, starting from index of 0
        threshold_val (float, optional): value of minimum threshold to determine the amount of attenuation at. Defaults to 0 (all data).
        gas_weight (float, optional): molar mass of gas to convert to molecules/cm2, only used for gas. Defaults to None.

    Returns:
        np.ndarray: sorted densities, fractions attenuated for each thickness/density value and corresponding errors, as well as 
        sorted initial flux values (before entering material) for each material thickness/density, to use mean of in commode analysis
    """
    

    data_filenames = []
    for file_path in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, file_path)):
            data_filenames.append(file_path)
    
    densities = []
    sums = []
    errors = []
    J_before_list = []
    for filename in data_filenames:
        search = re.search(thickness_search_string, filename)
        density = float(search.group(1))
        densities.append(density)

        if which_system == 'mulassis':
            df00 = mulassis_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                                   which_block=(before_after_layers[0],0))
            df20 = mulassis_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                                   which_block=(before_after_layers[1],0))
        if which_system == 'gras':
            df00 = gras_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                               which_boundary=before_after_layers[0])
            df20 = gras_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                               which_boundary=before_after_layers[1])


        # find histogram widths
        bins = df00['Elow'].to_list()
        width = df00['Eup'] - df00['Elow']


        ## find integral flux spectrum
        Jbefore_vals = df00['Fluence/flux'].values
        Jafter_vals = df20['Fluence/flux'].values
        
        # take reverse cumulative sum
        Jbefore_vals_rev = Jbefore_vals[::-1]
        Jafter_vals_rev = Jafter_vals[::-1]

        J0_rev = np.cumsum(Jbefore_vals_rev)
        J1_rev = np.cumsum(Jafter_vals_rev)


        IJ0 = J0_rev[::-1]
        IJ2 = J1_rev[::-1]

        # do same sum method for error values
        error_before_vals = df00['flux_error'].values
        error_after_vals = df20['flux_error'].values
        
        error_before_vals_rev = error_before_vals[::-1]
        error_after_vals_rev = error_after_vals[::-1]

        error_0_rev = np.cumsum(error_before_vals_rev)
        error_1_rev = np.cumsum(error_after_vals_rev)

        error_J0 = error_0_rev[::-1]
        error_J2 = error_1_rev[::-1]

        # find cumsum value for E ~= 49 MeV
        first_index_over_49 = next(i[0] for i in enumerate(bins) if i[1]>=threshold_val)

        IJ0_sum_past_threshold = IJ0[first_index_over_49]
        IJ2_sum_past_threshold = IJ2[first_index_over_49]

        error_J0_sum_past_threshold = error_J0[first_index_over_49]
        error_J2_sum_past_threshold = error_J2[first_index_over_49]

        # find fraction attenuated for flux and flux error
        sum_frac_attenuated = 1-(IJ2_sum_past_threshold/IJ0_sum_past_threshold)
        error_frac_attenuated = ((error_J2_sum_past_threshold/IJ2_sum_past_threshold)+
                                 (error_J0_sum_past_threshold/IJ0_sum_past_threshold)) * (IJ2_sum_past_threshold/IJ0_sum_past_threshold)

        # append to array for each thickness value
        J_before_list.append(IJ0_sum_past_threshold)
        sums.append(sum_frac_attenuated)
        errors.append(error_frac_attenuated)


    sums = np.array(sums)
    errors = np.array(errors)
    J_before_list = np.array(J_before_list)

    # for gas column density values, convert to molecules/m2    
    if gas_weight is not None:
        densities = np.array(densities) * (6.022e23/gas_weight)  # co2: 44.01
        densities *= (100**2)

    # sort from lowest thickness/column density to highest    
    sorted_densities = [i for i,j in sorted(zip(densities,sums))]
    sorted_fractions = [j for i,j in sorted(zip(densities,sums))]
    sorted_errors = [j for i,j in sorted(zip(densities,errors))]
    sorted_J_before_list = [j for i,j in sorted(zip(densities, J_before_list))]

    return [sorted_densities, sorted_fractions, sorted_errors, sorted_J_before_list]




def mulassis_frac_attenuated_over_energy(data_path:str, thickness:str, material:str, number_barriers:int, before_after_layers:tuple):
    """Calculate fraction attenuation per energy bin for one thickness and material. Only for MULASSIS and solid materials
    Args:
        data_path (str): path to data folder containing file
        thickness (str): thickness/column density of material, in cm
        material (str): name of material
        number_barriers (int): number of barriers in mulassis file, starting from index 0
        before_after_layers (tuple): which layers to select, starting from index 0

    Returns:
        list: [bins, width, sum_frac_attenuated, error_xvals, error_frac_attenuated] fraction attenuation for each
                energy bin
    """

    filename = f'{material}_vacuum_{thickness}_cm.txt'

    df00 = mulassis_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                            which_block=(before_after_layers[0],0))
    df20 = mulassis_reader(filename=f'{data_path}/{filename}', number_barriers=number_barriers, 
                            which_block=(before_after_layers[1],0))

    # find histogram widths
    bins = df00['Elow'].to_list()
    width = df00['Eup'] - df00['Elow']


    ## find integral flux spectrum
    Jbefore_vals = df00['Fluence/flux'].values
    Jafter_vals = df20['Fluence/flux'].values
    
    # perform reverse cumulative sum
    Jbefore_vals_rev = Jbefore_vals[::-1]
    Jafter_vals_rev = Jafter_vals[::-1]

    J0_rev = np.cumsum(Jbefore_vals_rev)
    J1_rev = np.cumsum(Jafter_vals_rev)

    IJ0 = J0_rev[::-1]
    IJ2 = J1_rev[::-1]

    # do same sum calculation on error values
    error_before_vals = df00['flux_error'].values
    error_after_vals = df20['flux_error'].values

    error_before_vals_rev = error_before_vals[::-1]
    error_after_vals_rev = error_after_vals[::-1]

    error_0_rev = np.cumsum(error_before_vals_rev)
    error_1_rev = np.cumsum(error_after_vals_rev)

    error_J0 = error_0_rev[::-1]
    error_J2 = error_1_rev[::-1]

    # find fraction attenuation for each energy value
    sum_frac_attenuated = 1-(IJ2/IJ0)
    error_frac_attenuated = ((error_J2/IJ2)+ (error_J0/IJ0)) * (IJ2/IJ0)


    error_xvals = (df00['Elow']+df00['Eup'])/2


    return [bins, width, sum_frac_attenuated, error_xvals, error_frac_attenuated]
    



def commode_effective_thickness(total_dust_fluence_per_size_bin:float, grain_size:float, 
                                mean_gcr_initial_integral_fluence:float):
    """Find the cumulative grain thickness that a spectrum of GCRs pass through
    normalized to the FOV of the SREM instrument (20 degrees^2)

    Args:
        total_dust_fluence_per_size_bin (float): total dust fluence from commode results for one grain size bin
        grain_size (float): grain size bin in meters
        mean_gcr_initial_integral_fluence (float): initial GCR fluence, taken the mean of 
                                                   all MULASSIS runs for one type of material

    Returns:
        float: cumulative thickness for one grain size, in cm
    """

    # FOV of SREM: 20 deg^2 --> sr
    steradian_normalization = (20*np.pi/180)**2 #= 0.1218

    # already in [m-2], convert 4pi sr to 20deg^2 FOV
    initial_gcr_integral_fluence = mean_gcr_initial_integral_fluence * steradian_normalization/(4*np.pi)

    grain_area = np.pi*((grain_size/2)**2) # m^2

    total_particles_per_gcr = total_dust_fluence_per_size_bin * grain_area

    # particles/GCR  * #GCRs/m2 (fluence)
    total_particles_for_all_GCRs = total_particles_per_gcr * initial_gcr_integral_fluence

    effective_thickness = total_particles_for_all_GCRs * grain_size * 100  # cm

    return effective_thickness # cm




def bar_plot(df_before:pd.DataFrame, df_after:pd.DataFrame, plot_type:str, spectra_df:pd.DataFrame=None):
    """Plot MULASSIS Histogram for either "default" (differential) or "integrated" versions.
    Can include overall GCR spectra line from SPENVIS. Dataframes must match those returned
    by mulassis_spectrum_reader and mulassis_reader functions.

    Args:
        df_before (pd.DataFrame): MULASSIS dataframe of GCR fluence before entering layer
        df_after (pd.DataFrame): MULASSIS dataframe of GCR fluence after material layer
        plot_type (str): "default", "differential", or "integral". If "default", will
                         plot histogram just as MULASSIS returns the data. If "differential"
                         will plot bin height/bin width. If "integral", will plot cumulative
                         sum from highest to lowest values.
        spectra_df (pd.DataFrame, optional): Dataframe returned by mulassis_spectrum_reader. If not None,
                                            plots initial GCR spectrum returned by SPENVIS. Defaults to None.
    """

    normfactor_angular = 0.25
    normfactor_spectrum = (4*np.pi*3.154e7)/(1e4)

    print('normfactor spectrum:', normfactor_spectrum)

    bins = df_before['Elow'].to_list()
    width = df_before['Eup'] - df_before['Elow']

    if plot_type == 'differential':
        print('plotting differential version')
        J0 = df_before['Fluence/flux'].values/width.values
        J1 = df_after['Fluence/flux'].values/width.values

        error_J0 = df_before['flux_error'].values/width.values
        error_J1 = df_after['flux_error'].values/width.values

    
    elif plot_type == 'integral':
        print('plotting integral version')
        Jbefore_vals = df_before['Fluence/flux'].values
        Jafter_vals = df_after['Fluence/flux'].values
        
        Jbefore_vals_rev = Jbefore_vals[::-1]
        Jafter_vals_rev = Jafter_vals[::-1]

        J0_rev = np.cumsum(Jbefore_vals_rev)
        J1_rev = np.cumsum(Jafter_vals_rev)

        J0 = J0_rev[::-1]
        J1 = J1_rev[::-1]

        error_before_vals = df_before['flux_error'].values
        error_after_vals = df_after['flux_error'].values
        
        error_before_vals_rev = error_before_vals[::-1]
        error_after_vals_rev = error_after_vals[::-1]

        error_0_rev = np.cumsum(error_before_vals_rev)
        error_1_rev = np.cumsum(error_after_vals_rev)

        error_J0 = error_0_rev[::-1]
        error_J1 = error_1_rev[::-1]

    elif plot_type == 'default':
        print('plotting default version')
        J0 = df_before['Fluence/flux']
        J1 = df_after['Fluence/flux']

        error_J0 = df_before['flux_error']
        error_J1 = df_after['flux_error']

    else:
        print('plot type not known')
        exit()
    
    fillcolor = 'forestgreen'
    fillerrorcolor = 'limegreen'
    plt.bar(bins, height=J1, width=width, align='edge', fill=True, #*1e4/(4*np.pi*3.154e7)
            color=fillcolor, edgecolor=fillcolor, label='After Layer')
    
    plt.bar(bins, height=J0, width=width, align='edge', fill=False,  #*1e4/(np.pi*3.154e7)
            edgecolor='k', label='Before Layer')
    
    error_xvals = (df_before['Elow']+df_before['Eup'])/2
    plt.errorbar(error_xvals, J0, yerr=error_J0, fmt='none', color='k')
    plt.errorbar(error_xvals, J1, yerr=error_J1, fmt='none', color=fillerrorcolor)

    
    if spectra_df is not None:
        if plot_type == 'differential':
             plt.plot(spectra_df['Energy_MeV_n'],
                    spectra_df['DFlux_m2_sr_s_MeV_n']*normfactor_angular*normfactor_spectrum, 
                    label='Diff Spectrum x $(4\pi 3.15e7)/(1e4)$ x (1/4)', color='red')       
        elif plot_type == 'integral':
            plt.plot(spectra_df['Energy_MeV_n'], 
                     spectra_df['IFlux_m2_sr_s']*normfactor_angular*normfactor_spectrum, 
                     label='Int Spectrum x angular_normfactor x $(4\pi 3.15e7)/(1e4)$', color='red')
        else:
            print('spectrum plot type not known')
            exit()




def thickness_interpolation(all_thicknesses:np.array, all_fractions_attenuated:np.array, frac_intersection:int=0.08):
    """Interpolate the material thickness vs. fraction attenuation curve for one material
    Use data returned from mulassis_frac_atten_return() definition

    Args:
        all_thicknesses (np.array): array of all thickness values for one grain material
        all_fractions_attenuated (np.array): array of all fractions attenuated, in line with thickness values
        frac_intersection (int): attenuation fraction value desired to find thickness at. Default 0.08

    Returns:
        (arr, arr, float): the new interpolated fraction attenuated and thickness values (used for plotting), 
                            and the thickness value at the desired intersection point
    """
    thick_new = np.linspace(0.01,10,1000)

    func = interpolate.interp1d(all_thicknesses, all_fractions_attenuated)
    frac_new = func(thick_new)
    
    
    from scipy.optimize import curve_fit
    def func(x, a, b, c, d):
        return a*np.log(b*(x + c)) + d
    

    popt, pcov = curve_fit(func, thick_new, frac_new,  maxfev=5000)
    frac_new = func(thick_new, *popt)
    

    eight_percent = frac_intersection*np.ones(len(thick_new))

    idx = np.argwhere(np.diff(np.sign(eight_percent-frac_new))).flatten()
    intersection_thickness = thick_new[idx]
    intersection_fraction = frac_new[idx]

    print( "Parameters from least-squares fit:")
    print( "a =", popt[0], "+/-", pcov[0,0]**0.5)
    print( "b =", popt[1], "+/-", pcov[1,1]**0.5)
    print( "c =", popt[2], "+/-", pcov[2,2]**0.5)
    print( "d =", popt[3], "+/-", pcov[3,3]**0.5)
    print('INTERSECTION THICKNESS:', intersection_thickness)
    print('============================================')

    return frac_new, thick_new, intersection_thickness




def commode_intersection_interpolation(size_bins:np.array, cumulative_thickness:np.array, intersection_vals:list=[0.7, 1.36, 2.02]):
    """Interpolate the material thickness vs. fraction attenuation curve for one material
    Use data returned from mulassis_frac_atten_return() definition

    Args:
        size_bins (np.array): array of commode grain size bins to be interpolated over (in um)
        cumulative_thickness (np.array): array of cumulative thickness values for each size bins (in cm)
        intersection_vals (int): MULASSIS thickness values to find the grain size bin intersection point at.
                                 Defaults to 1.36 +/- 0.66 cm.

    Returns:
        (arr, arr, list): the new interpolated cumulative thickness and grain bins (used for plotting), 
                            and the thickness value at the desired intersection points
    """

    bins_new = np.linspace(0.1,2000,int(2e4))

    func = interpolate.interp1d(size_bins, cumulative_thickness, fill_value='extrapolate')
    cumulative_thickness_new = func(bins_new)

    
    intersection_bins = []
    for i in intersection_vals:
        mulassis_intersection = i*np.ones(len(bins_new))

        idx = np.argwhere(np.diff(np.sign(mulassis_intersection-cumulative_thickness_new))).flatten()
        if len(idx) >= 1:
            intersection_bin = bins_new[idx][0]
            intersection_bins.append(intersection_bin)
        else:
            intersection_bins.append(np.nan)

    # print( "Parameters from least-squares fit:")
    # print( "a =", popt[0], "+/-", pcov[0,0]**0.5)
    print('INTERSECTION BIN:', intersection_bins)
    print('============================================')
    return cumulative_thickness_new, bins_new, intersection_bins




def save_commode_results(trajectory_folder_name:str):
    """Saves commode results for one coma and one trajectory as Numpy array.
    Determines the cumulative thickness for each grain bin size and saves to 
    an array.

    Args:
        trajectory_folder_name (str): folder name for all bin size results for one commode trajectory

    Returns:
        np.array: array containing the grain size bins (in um), the total fluence and cumulative thickness value (in cm)
                  for each bin.
    """
    
    # save grain sizes to array
    f = open(f"commode_results/{trajectory_folder_name}/header_trajectory.txt", "r")
    line_number = 0
    for line in f:
        values = re.split(', |\n|: ', line)
        if line_number == 5:
            gs = values[1:-1]
        line_number +=1
    f.close()
    grain_sizes = np.array([float(i) for i in gs])

    # save total/maximum fluence per grain size to array
    total_fluences = []
    for traj_number, gs in zip(range(len(grain_sizes)), grain_sizes):
        path = f'commode_results/{trajectory_folder_name}/dust_density_along_trajectory{traj_number}.txt'
        cols = ['dust_density', 'flux', 'fluence', 'R', 'S', 'time']
        data = pd.read_csv(path, names=cols, sep=', ', engine='python')#, skiprows=11)

        total_fluence = np.max(data['fluence'])
        total_fluences.append(total_fluence)
    total_fluences = np.array(total_fluences)
    

    # load in mean initial GCR fluence
    material = 'enstatite'
    frac_atten_return_path = f'MULASSIS_tables/mulassis_width_files/fraction_attenuation_return/return_{material}.npy'
    frac_atten_return = np.load(frac_atten_return_path)
    gcr_fluence = frac_atten_return[3]
    mean_gcr_fluence = np.mean(gcr_fluence)

    # find effective thickness, save as array
    eff_thick = []
    for dust_fluence, grain_size in zip(total_fluences, grain_sizes):
        effective_thickness = commode_effective_thickness(total_dust_fluence_per_size_bin=dust_fluence,
                                                        grain_size=grain_size,
                                                        mean_gcr_initial_integral_fluence=mean_gcr_fluence)
        eff_thick.append(effective_thickness)
    eff_thick = np.array(eff_thick)

    # save grain sizes, total fluence, and effective thickness into one 3x#bins array
    grain_size_fluence_thickness = np.array([grain_sizes, total_fluences, eff_thick])
    np.save(f'commode_results/{trajectory_folder_name}_grainsizes_max_fluences_thicknesses.npy', grain_size_fluence_thickness)

    return grain_size_fluence_thickness

