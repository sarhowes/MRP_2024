import matplotlib.pyplot as plt
import matplotlib as mpl
from definitions_final import *
from matplotlib.pyplot import cm


# universal plotting formatting
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = '14.0'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = 7.5,6


# ========= plot MULASSIS histograms =========
material_name = 'enstatite'
thickness = '1e-6'
plot_type_str = 'integral'
spectra_path = 'spenvis_gcr_spectra_tables/gcr_z1_flux_spectra.txt'

def mulassis_histogram_plotter(material_name:str, thickness:str, plot_type_str:str, spectra_path=None, save_as=None):
    """Plot and save MULASSIS histogram

    Args:
        material_name (str): name of material to plot
        thickness (str): thickness of material to plot
        plot_type_str (str): "integral", "differential", or "default"
        spectra_path (str, optional): Path of SPENVIS spectra, if want to plot. Defaults to None.
        save_as (str, optional): Type of file to save. If None, then will not save plot, just show. Defaults to None.
    """
    # =========== read in spectrum files ================
    if spectra_path is not None:
        spec_h = mulassis_spectrum_reader(spectra_path)
    else:
        spec_h = None
    # ===================== PLOT HISTOGRAM FOR ONE FILE =====================

    mulassis_path = f'MULASSIS_tables/mulassis_width_files'
    mulassis_subpath = f'mulassis_{material_name}_vacuum_z1_1mev_1e5mev_omni'
    mulassis_file = f'{mulassis_path}/{mulassis_subpath}/{material_name}_vacuum_{thickness}_cm.txt'
    mulassis_00 = mulassis_reader(filename=mulassis_file, number_barriers=3, which_block=(0,0))
    mulassis_10 = mulassis_reader(filename=mulassis_file, number_barriers=3, which_block=(1,0))



    bar_plot(mulassis_00, mulassis_10, plot_type=plot_type_str, spectra_df=spec_h)
    plt.axvline(49, color='r', ls='--', label='49 MeV')
    plt.legend(loc='lower left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (MeV/n)')
    plt.ylabel('Proton Fluence (cm$^{-2}$)')
    plt.title(f'MULASSIS Omnidirectional {plot_type_str}\n Proton Histogram for {float(thickness)} cm {material_name}')

    if save_as is not None:
        plt.savefig(f'MULASSIS_histogram_figures/{plot_type_str}_{material_name}_{thickness}_cm.{save_as}', dpi=300)
    else:
        plt.show()


# mulassis_histogram_plotter(material_name, thickness, plot_type_str, save_as='pdf')


# ========= plot MULASSIS fraction attenuation over energy =========
material_savenames = ['h2o_ice', 'amorphous_carbon', 'graphite', 'olivine_fe', 'enstatite',  'pyroxene_mg', 'troilite']
material_labels = ['H$_2$O ice', 'C (amorphous)', 'C (graphite)', 'Olivine (Fe)','Enstatite', 'Pyroxene (Mg)',  'Troilite']
material_savenames = material_savenames[::-1]
material_labels = material_labels[::-1]

thickness_vals = ['4.64158883', '2.15443469',  '1.0', '0.46415888',  '0.21544347',  '1e-1',  '1e-2']
thickness_labels = ['4.6 cm', '2.2 cm', '1.0 cm', '0.46 cm', '0.22 cm',  '0.1 cm', '0.01 cm']



def mulassis_frac_atten_energy_hist_plotter(which_type:str, one_thickness:float=None, material_list:list=None, material_labels:list=None,
                                            one_material:str=None, thickness_list:list=None, thickness_labels:list=None, save_as:str=None):
    """plot fraction attenuation over all (integrated) energy bins. Can either plot one thickness value
    over all material types, or one material type over all thickness values.

    Args:
        which_type (str): if 'over_materials': will plot all materials for one thickness value. one_thickness must be filled in.
                          if 'over_thickness': will plot all thicknesses for oen material. one_material must be filled in.
        
        one_thickness (float, optional): Single thickness value for 'over_materials' option. Defaults to None.
        material_list (list, optional): List of materials to plot for 'over_materials' option. Defaults to None.
        material_labels (list, optional): list of material labels for 'over_materials' option. Defaults to None.

        one_material (str, optional): Single material for 'over_thickness' option. Defaults to None.
        thickness_list (list, optional): List of thicknesses to plot over for 'over_thickness' option. Defaults to None.
        thickness_labels (list, optional): List of thickness labels to plot over for 'over_thickness' option. Defaults to None.
        
        save_as (str, optional): Type of file to save as ('pdf' or 'png'...) If None, will just show the plot without saving. 
                                 Defaults to None.

    """



    # plot one thickness over all materials
    if which_type == 'over_materials':
        color = cm.turbo_r(np.linspace(0, 1, len(material_list)))
        for material,lab,c in zip(material_list, material_labels,color):
            material_path = f'MULASSIS_tables/mulassis_width_files/mulassis_{material}_vacuum_z1_1mev_1e5mev_omni'

            frac_atten_energy_return = mulassis_frac_attenuated_over_energy(data_path=material_path, thickness=one_thickness, 
                                                                            material=material, number_barriers=3, 
                                                                            before_after_layers=(0,1))


            bins = frac_atten_energy_return[0]
            width = frac_atten_energy_return[1]
            sum_frac_attenuated = frac_atten_energy_return[2]

            plt.bar(bins, height=sum_frac_attenuated, width=width, align='edge', fill=True,
                color=c, edgecolor='k', alpha=0.5, label=lab)


        plt.xscale('log')
        plt.axhline(0.08, ls='--', color='k')
        plt.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)
        plt.ylim(0.0,0.2)
        plt.title(f'Fraction Attenuation for {one_thickness} cm')
        plt.xlabel('Energy (MeV/n)')
        plt.ylabel('Fraction Attenuated')
        plt.legend()
        if save_as is not None:
            plt.savefig(f'mulassis_histogram_figures/frac_atten_over_energy_all_materials_{one_thickness}_cm.{save_as}', dpi=300)
        else:
            plt.show()
        plt.close()

    # plot one material over a range of thicknesses
    elif which_type == 'over_thickness':
        color = cm.viridis(np.linspace(0, 1, len(thickness_list)))
        for thickness,lab,c in zip(thickness_list, thickness_labels, color):

            material_path = f'MULASSIS_tables/mulassis_width_files/mulassis_{one_material}_vacuum_z1_1mev_1e5mev_omni'

            frac_atten_energy_return = mulassis_frac_attenuated_over_energy(data_path=material_path, thickness=thickness, 
                                                                            material=one_material, number_barriers=3, 
                                                                            before_after_layers=(0,1))


            bins = frac_atten_energy_return[0]
            width = frac_atten_energy_return[1]
            sum_frac_attenuated = frac_atten_energy_return[2]

            plt.bar(bins, height=sum_frac_attenuated, width=width, align='edge', fill=True,
                    color=c, edgecolor='k', alpha=0.5, label=lab)


        plt.xscale('log')
        plt.axhline(0.08, ls='--', color='k', label='8% Attenuation')
        plt.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)
        # plt.ylim(0.0,0.2)
        plt.title(f'Fraction Attenuation for {one_material}')
        plt.xlabel('Energy (MeV/n)')
        plt.ylabel('Fraction Attenuated')
        plt.legend()
        if save_as is not None:
            plt.savefig(f'mulassis_histogram_figures/frac_atten_over_all_thicknesses_{one_material}.{save_as}')
        else:
            plt.show()
        plt.close()
    
    else:
        print('which type not known')
        exit()



# mulassis_frac_atten_energy_hist_plotter(which_type='over_materials', one_thickness=1.0, material_list=material_savenames, 
#                                         material_labels=material_labels, save_as='pdf')


# mulassis_frac_atten_energy_hist_plotter(which_type='over_thickness', one_material='troilite', thickness_list=thickness_vals, 
#                                         thickness_labels=thickness_labels, save_as='pdf')




# ========= plot MULASSIS fraction attenuation =========
material_names = ['co2_vapor', 'co_vapor',  'h2o_vapor']
material_labels = ['CO$_2$', 'CO', 'H$_2$O' ]
vapor_weights = [44.01, 28.01, 18.01528]



def mulassis_frac_atten_plotter(material_savenames:list, material_labels:list, calc_frac_atten:bool=False, 
                                interpolate:bool=False, save_as:str=None):
    """Plot and save MULASSIS fraction attenuation plot, both in log and linear scale. Can choose to
    interpolate data to find average 8% thickness value. 

    Args:
        material_savenames (list): list of material names, as appear in filenames
        material_labels (list): list of material names in same order as material_savenames,
                                but formatted for labels
        calc_frac_atten (bool, optional): If True, will run frac_attenuated_per_thickness function
                                          and save data to fraction_attenuation_return folder
        interpolate (bool, optional): If True, will calculate interpolation to find
                                      8% intersection. Defaults to False.
        save_as (str, optional): Format of savefile. If None, will just show plot. Defaults to None.

    """

    color = cm.turbo_r(np.linspace(0, 1, len(material_savenames)))
    fig, (ax1, ax2) = plt.subplots(2, figsize=(7.5,6), sharey=True)


    intersection_thicknesses = []
    # loop over all material name files, plot fraction attenuated
    for material,c,lab in zip(material_savenames, color, material_labels):
        material_path = f'MULASSIS_tables/mulassis_width_files/mulassis_{material}_vacuum_z1_1mev_1e5mev_omni'
        print(material_path)

        if calc_frac_atten == False:
            frac_atten_return = np.load(f'MULASSIS_tables/mulassis_width_files/fraction_attenuation_return/return_{material}.npy')
        else:
            frac_atten_return = frac_attenuated_per_thickness(data_path=material_path, density_search_string=f'{material}_vacuum_(.*)_cm', 
                                                            which_system='mulassis', number_layers=3, before_after_layers=(0,1), 
                                                            threshold_val=49)
            np.save(f'MULASSIS_tables/mulassis_width_files/fraction_attenuation_return/return_{material}.npy', frac_atten_return)
        

        thicknesses = frac_atten_return[0]
        frac = frac_atten_return[1]
        frac_err = frac_atten_return[2]


        # ### interpolate to find intersection with 8% line
        if interpolate == True:
            frac_new, thick_new, intersection_thickness = thickness_interpolation(thicknesses[4:-1], frac[4:-1], frac_intersection=0.08)
            intersection_thicknesses.append(intersection_thickness)
            
            ax1.plot(thick_new, frac_new, color=c, ls='-', lw=2)
            ax2.plot(thick_new, frac_new, color=c, ls='-', lw=2)

            ax1.errorbar(thicknesses[4:-1], frac[4:-1], yerr=frac_err[4:-1],
                        fmt='o', label=lab, color=c)
            ax2.errorbar(thicknesses[4:-1], frac[4:-1], yerr=frac_err[4:-1],
                        fmt='o', color=c)

        else:
            ax1.errorbar(thicknesses, frac, yerr=frac_err,
                        fmt='o-', label=lab, color=c)
            ax2.errorbar(thicknesses, frac, yerr=frac_err,
                        fmt='o-', color=c)


        
    frac_intersection = 0.08

    if interpolate == True:

        ax1.scatter(intersection_thicknesses, frac_intersection*np.ones(len(intersection_thicknesses)), 
                    marker='x', color='r', s=50, lw=2.5, zorder=10, label='$8\%$ line intersection')
        ax2.scatter(intersection_thicknesses, frac_intersection*np.ones(len(intersection_thicknesses)), 
                    marker='x', color='r', s=50, lw=2.5, zorder=10)

        ax1.set_title(f'mean thickness: {np.mean(intersection_thicknesses).round(4)}$\pm${np.std(intersection_thicknesses).round(2)} cm')

        ax2.set_ylim(0,0.2)
        ax2.set_xlim(0.7,3)

        ax1.set_xlim(0.7, 3)
        ax1.set_ylim(0,0.2)



    ax1.set_xscale('log')

    ax1.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)
    ax2.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)

    plt.ylabel(' ')
    plt.xlabel('Thickness (cm)')
    fig.text(0.01, 0.5, 'Fraction Attenuated', va='center', rotation='vertical')

    ax1.axhline(frac_intersection, color='k', ls='--', label='8% Attenuation')
    ax2.axhline(frac_intersection, color='k', ls='--')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
   
    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95,0.5))


    if save_as is not None:
        if interpolate == False:
            plt.savefig(f'MULASSIS_frac_atten_figures/frac_attenuated_errorbar_forthesis.{save_as}', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'MULASSIS_frac_atten_figures/frac_attenuated_interpolated_forthesis.{save_as}', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    else:
        plt.show()


# mulassis_frac_atten_plotter(material_savenames, material_labels, calc_frac_atten=False, interpolate=False, save_as='pdf')




# ========= plot commode cumulative thickness =========

main_path_twocases = 'two_best_scenarios'
commode_traj_foldernames_twocases = ['powerlaw_3.5',
                            'k3.4_Qd1e3_Qg1111',
                            'k3.3_Qd1e3_Qg1250',
                            'k3.2_Qd1e3_Qg1429',
                            'k3.1_Qd1e3_Qg1667',

                            'k4.1_Qd2e3_Qg1e3', 
                            'k4.2_Qd3e3_Qg1e3', 
                            'k4.3_Qd4e3_Qg1e3', 
                            'k4.4_Qd5e3_Qg1e3', 
                            'k4.5_Qd6e3_Qg1e3']               
labelnames_twocases = ['$\kappa=3.5$, $Q_d=1\\times10^3$, $\chi=1.0$',
              '$\kappa=3.4$, $Q_d=1\\times10^3$, $\chi=0.9$',
              '$\kappa=3.3$, $Q_d=1\\times10^3$, $\chi=0.8$',
              '$\kappa=3.2$, $Q_d=1\\times10^3$, $\chi=0.7$',
              '$\kappa=3.1$, $Q_d=1\\times10^3$, $\chi=0.6$\n',

              '$\kappa=4.1$, $Q_d=2\\times10^3$, $\chi=2.0$',
              '$\kappa=4.2$, $Q_d=3\\times10^3$, $\chi=3.0$',
              '$\kappa=4.3$, $Q_d=4\\times10^3$, $\chi=4.0$',
              '$\kappa=4.4$, $Q_d=5\\times10^3$, $\chi=5.0$',
              '$\kappa=4.5$, $Q_d=6\\times10^3$, $\chi=6.0$']
              
              
main_path_powerlaw =  'power_law'
commode_traj_powerlaw_barvals = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
commode_traj_powerlaw_filenames = ['powerlaw_3.1', 'powerlaw_3.2', 
                                   'powerlaw_3.3', 'powerlaw_3.4', 
                                   'powerlaw_3.5', 'powerlaw_3.6', 
                                   'powerlaw_3.7', 'powerlaw_3.8', 
                                   'powerlaw_3.9', 'powerlaw_4.0', 
                                   'powerlaw_4.1', 'powerlaw_4.2', 
                                   'powerlaw_4.3', 'powerlaw_4.4', 
                                   'powerlaw_4.5']



main_path_dpr = 'dust_production_rate_log'
commode_traj_dpr_barvals = [1e3, 1.6e3, 2.5e3, 4e3, 6.3e3, 1e4, 1.6e4, 2.5e4, 4e4]
commode_traj_dpr_filenames = ['dpr_1e3', 'dpr_1.6e3', 
                              'dpr_2.5e3', 'dpr_4e3',
                              'dpr_6.3e3','dpr_1e4',
                              'dpr_1.6e4', 'dpr_2.5e4',
                              'dpr_4e4']


main_path_chi = 'chi'
commode_traj_chi_barvals = [0.73, 1.58, 2.44, 3.29, 4.15, 5]
commode_traj_chi_filenames = ['Qg_6849_chi_0.73', 'Qg_3165_chi_1.58', 
                                'Qg_2049_chi_2.44', 'Qg_1520_chi_3.29', 
                                'Qg_1205_chi_4.15', 'Qg_1000_chi_5']


# for individual parameters, with color bar
def commode_cumulative_thickness_colorbar_plotter(main_path:str, commode_traj_filenames:list, barvals:list,
                                                  which_label:str, calc_cumulative_thickness:bool=False,
                                                  find_intersections:bool=False, save_as:str=None):
    """Plot the cumulative thickness curves from ComMoDE results for power-law index (kappa), 
    dust production rate (dpr), and dust-to-gas ratio (chi) changes. Input the path, list of values that
    are changed, and list of filenames to make plot with colorbar. Can also plot values for intersection
    with MULASSIS total thickness prediciton + upper/lower bounds.

    Args:
        main_path (str): name/path of folder containing all commode results
        commode_traj_filenames (list): list of names of all commode results within folder
        barvals (list): list of values that are changed within commode analysis
        which_label (str): 'kappa' for powerlaw index analysis, 'dpr' for dust production rate analysis,
                           'chi' for dust-to-gas ratio analysis
        calc_cumulative_thickness (bool, optional): If True, performs the save_commode_results() function.
                                                    If False, then assumes that commode results are already 
                                                    saved and will load them instead. Defaults to False.
        find_intersections (bool, optional): If True, finds the intersection with MULASSIS thickness prediction and 
                                            plots the results. Defaults to False.
        save_as (str, optional): 'pdf' to save plots as PDF, 'png' to save as png image. If None, then will just
                                show the plot without saving. Defaults to None.
    """


    upper_intersection = []
    center_intersection = []
    lower_intersection = []

    all_gs_bins = []
    all_effthicks = []

    for folder in commode_traj_filenames:
        filename = f'{main_path}/traj2_{folder}'
        print(filename)


        if calc_cumulative_thickness == True:
            grainsize_fluence_thickness = save_commode_results(filename)
        else:
            grainsize_fluence_thickness = np.load(f'commode_results/{filename}_grainsizes_max_fluences_thicknesses.npy')
        
        eff_thick_cumsum = np.cumsum(grainsize_fluence_thickness[2][::-1])
        eff_thick_cumsum = eff_thick_cumsum[::-1]
        grainsize_bins = grainsize_fluence_thickness[0]*1e6
        
        all_gs_bins.append(grainsize_bins)
        all_effthicks.append(eff_thick_cumsum)

        if find_intersections == True:
            cumulative_thickness_new, size_bins, intersection_bins = commode_intersection_interpolation(grainsize_bins, 
                                                                                                        eff_thick_cumsum,
                                                                                                        [0.7, 1.36, 2.02])

            upper = intersection_bins[2]
            if upper is np.nan:
                upper_intersection.append(intersection_bins[1])
            else:
                upper_intersection.append(upper)

            center_intersection.append(intersection_bins[1])

            lower = intersection_bins[0]
            if lower is np.nan:
                lower_intersection.append(intersection_bins[1])
            else:
                lower_intersection.append(lower)



    all_effthicks = np.array(all_effthicks).T
    all_gs_bins = np.array(all_gs_bins).T
    data_lines = np.array([all_gs_bins, all_effthicks]).T

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(7.5,6))

    if which_label == 'kappa':
        line_collection = LineCollection(data_lines, array=barvals, cmap="turbo", lw=2.5)
        plt.colorbar(line_collection, label="Power-Law Exponent ($\kappa$)")
        
    elif which_label == 'dpr':
        norm = LogNorm(1e3, 4e4)
        line_collection = LineCollection(data_lines, array=barvals, cmap="turbo", lw=2.5, norm=norm)
        plt.colorbar(line_collection, label='Dust Production Rate [kg s$^{-1}$]')

    elif which_label == 'chi':
        line_collection = LineCollection(data_lines, array=barvals, cmap="turbo", lw=2.5)
        plt.colorbar(line_collection, label='Dust-to-Gas Ratio ($\chi$)')
    else:
        print('label not known')
        exit()
    
    ax.add_collection(line_collection)

    # for 8% attenuation measured from mulassis, 1.36 +/- 0.66 cm
    plt.axhspan(0.70, 2.02, color='green', alpha=0.3)
    plt.axhline(1.36, color='green', ls='--', alpha=0.8, label='MULASSIS Estimation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Grain size bin [$\mu$m]')
    plt.ylabel('Cumulative Thickness [cm]')
    plt.legend(loc='lower right')
    plt.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)


    if save_as is not None:
        plt.savefig(f'commode_figures/commode_cumulative_thickness_{which_label}_.{save_as}', bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # plot intersections
    if find_intersections == True:
        print('upper intersections:', upper_intersection)
        print('center intersections:', center_intersection)
        print('lower intersections', lower_intersection)

        barvals = np.array(barvals)
        plt.plot(barvals, center_intersection, marker='o', color='green', ls='--')
        plt.fill_between(barvals, lower_intersection, upper_intersection, color='green', alpha=0.3)
        plt.ylabel('Intersection grain size bin [$\mu$m]')
        plt.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)

        if which_label == 'dpr':
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Dust Production Rate [kg s$^{-1}$]')
        elif which_label == 'chi':
            plt.xlabel('Dust-to-Gas Ratio ($\chi$)')

        elif which_label == 'kappa':
            plt.xlabel('Power-law exponent ($\kappa$)')
        else:
            print('plot type not known')
            exit()


        if save_as is not None:
            plt.savefig(f'commode_figures/commode_cumulative_thickness_intersection_vals_{which_label}.{save_as}', dpi=300, bbox_inches='tight')
        else:
            plt.show()




# for two cases, without color bar
def commode_cumulative_thickness_plotter(main_path, commode_traj_foldernames, labelnames, 
                                         calc_cumulative_thickness=False, save_as=None):

    colors = cm.seismic(np.linspace(0, 1, len(commode_traj_foldernames)))


    fig, ax = plt.subplots(figsize=(7.5,6))

    for folder,c,l in zip(commode_traj_foldernames, colors, labelnames):
        print(folder)
        if calc_cumulative_thickness == True:
            grainsize_fluence_thickness = save_commode_results(f'{main_path}/traj2_{folder}')
        else:
            grainsize_fluence_thickness = np.load(f'commode_results/{main_path}/traj2_{folder}_grainsizes_max_fluences_thicknesses.npy')
        
        eff_thick_cumsum = np.cumsum(grainsize_fluence_thickness[2][::-1])
        eff_thick_cumsum = eff_thick_cumsum[::-1]
        grainsize_bins = grainsize_fluence_thickness[0]*1e6

        ax.plot(grainsize_bins, 
                eff_thick_cumsum, 
                color=c, lw=2.5,
                label=l)


    plt.axhspan(0.70, 2.02, color='green', alpha=0.3)
    plt.axhline(1.36, color='green', ls='--', alpha=0.8, label='MULASSIS Estimation')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Grain size bin [$\mu$m]')
    plt.ylabel('Cumulative Thickness [cm]')
    # plt.legend(loc='lower right')

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95,0.5))

    plt.grid(visible='True', which='major', axis='both', color='gray', alpha=0.4)

    if save_as is not None:
        plt.savefig(f'commode_figures/commode_cumulative_thickness_{main_path}.{save_as}', bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.show()



# commode_cumulative_thickness_colorbar_plotter(main_path=main_path_powerlaw, commode_traj_filenames=commode_traj_powerlaw_filenames,
#                                               barvals=commode_traj_powerlaw_barvals, which_label='kappa', calc_cumulative_thickness=False,
#                                               find_intersections=True, save_as='pdf')

# commode_cumulative_thickness_colorbar_plotter(main_path=main_path_dpr, commode_traj_filenames=commode_traj_dpr_filenames,
#                                               barvals=commode_traj_dpr_barvals, which_label='dpr', calc_cumulative_thickness=False,
#                                               find_intersections=True, save_as='pdf')

# commode_cumulative_thickness_colorbar_plotter(main_path=main_path_chi, commode_traj_filenames=commode_traj_chi_filenames,
#                                               barvals=commode_traj_chi_barvals, which_label='chi', calc_cumulative_thickness=False,
#                                               find_intersections=True, save_as='pdf')

# commode_cumulative_thickness_plotter(main_path_twocases, commode_traj_foldernames_twocases, labelnames_twocases, 
#                                          calc_cumulative_thickness=False, save_as='pdf')