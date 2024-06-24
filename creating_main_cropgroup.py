# CROP TYPE DATAFRAMES FOR THE NEW MODEL

#Imports
from re import X
from types import LambdaType
from geopandas import geodataframe
from geopandas.geodataframe import _dataframe_set_geometry
from geopandas.plotting import plot_dataframe
import numpy as np
from numpy.lib.arraysetops import isin
from pandas.core.indexes.numeric import Int64Index
from scipy.integrate import odeint
from scipy.integrate import quad
import pandas as pd
import geopandas as gpd
import os
import fiona as fiona
import seaborn as sns


import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

#
import os
import sys

#


'''

IN THIS VERSION I HAVE PUT IN THE CARBON OF MANURE THAT IS SUGGESTED BY THE VERA DATABASE


'''
global VERA_database_amount_of_carbon_in_manure_per_hectar_total
VERA_database_amount_of_carbon_in_manure_per_hectar_total = 300 


global Calculated_pasture_added_carbon
Calculated_pasture_added_carbon = 0


def input_2020():
        ''''''

        ''' First introduction of import of the BLOCK DATA'''
        global df_input_1512_2020
        
        df_input_1512_2020 = gpd.read_file(r' # Path to shapefile with the interesting fields, where different crops are assigned a value')

        global df_input_2020
        df_input_2020 = pd.DataFrame(df_input_1512_2020)

input_2020()


def winterwheat_import():
        ''''''

        #The basefile will be the original df which all other will have the same shape as. I will start with winterwheat


        df_winterwheat = df_input_1512_2020[df_input_1512_2020['GRODKOD'] == '4'].copy()
        df_adding_2 = df_input_1512_2020[df_input_1512_2020['GRODKOD'] == '1']
        df_adding_3 = df_input_1512_2020[df_input_1512_2020['GRODKOD'] == '7'] 
        df_adding_4 = df_input_1512_2020[df_input_1512_2020['GRODKOD'] == '8'] 
        df_winterwheat = pd.concat([df_winterwheat, df_adding_2], axis=0)
        df_winterwheat = pd.concat([df_winterwheat, df_adding_3], axis=0)
        df_winterwheat = pd.concat([df_winterwheat, df_adding_4], axis=0)

        
        df_winterwheat.drop(columns= ['KUND_LAN','OBJECTID','SKIFTESBET','geometry'], inplace=True, axis=1)

        #Winter Cereals
        #1,4,7,8

        print(df_winterwheat)
        #print(df_winterwheat.info())

        df_basefile_input = pd.read_excel(r'')

        df_base = pd.DataFrame(df_basefile_input)

        df_work = pd.concat([df_winterwheat, df_base], axis="columns")
      
        global df_working_crop

        df_working_crop = pd.DataFrame(df_work).copy()
        df_working_crop.drop(columns =['fields_nr'], inplace= True)



        ''' Below is the classification and decisons made on "Carbon basis" - how much NPP is produced and where in the system the carbon goes
        This is from Swedish national data and is not something that directly can be applied without understanding the dynamics and possible local factors'''
        

        def Scenario_factors_for_carbon_flow_analysis():

           
                
            ''''''

            carbon_in_fodder_import = 100        

            barged_factor = 0.33                                    
            barged_factor_legumes = 0.04
            
            
            left_to_soil_factor = 1-barged_factor                   

            fodder_constant_from_harvest = 0.35                               
            fodder_constant_from_harvest_rapeseed = 0.67
            fodder_constant_from_harvest_legumes = 0.8                         
            fodder_constant_from_harvest_ley = 0.73
            fodder_constant_from_harvest_other = 0
            
            harvest_out_constant = 1 -fodder_constant_from_harvest            
            harvest_out_constant_rapeseed = 1 -fodder_constant_from_harvest_rapeseed            
            harvest_out_constant_legumes = 1 -fodder_constant_from_harvest_legumes           
            harvest_out_constant_ley = 1 -fodder_constant_from_harvest_ley          
            harvest_out_constant_other = 1 
            
            #Carbon out group
            
            c_food_out_human =  0.35          #Marknadsöversikt spannmål 2014
            
            c_export_out =  0.4               #Marknadsöversikt spannmål 2014
            
            c_industry_out =  0.25            #Marknadsöversikt spannmål 2014
            
            
            ''' SCENACRIO BESLUT TAS MED DESSA '''
            
            #Residues groupe
            
            residues_biogas_factor = 0.20                          # in the BAU scenario
                                                                # This can change based on what we want in scenario 
                                                                
            residues_bioCHAR_factor = 0                         # in the BAU scenario
                                                                # This can change based on what we want in scenario    
                                                                
                                                                
                                                                
            residue_carbon_to_animal = 0.24          # SCB2012
            
            residue_carbon_to_animal_rapeseed = 0.01
            
            
            residue_carbon_to_animal_fodder = 0.01    # SCB2012
            
            residue_carbon_to_animal_fodder_legumes = 0.01
            
            
            
            decay_rate = 0.758        # for the wwather as used in peopley bolinder iDBM modle 
            
            Re = 0.8  # values from ICBM papers 
            
            
            
            # Humification groupes
            
            hum_factor_bio_manure = 0.40            # Assumtion as of now 
            
            hum_factor_bioCHAR_manure = 1           # Assume that all biochar that is produced is stable for a rather long time

    
            ICBM_total_factor_manure = 0.32 * decay_rate*Re
            
            ICBM_total_factor_straw = 0.2 *decay_rate*Re
            
            ICBM_total_factor_root = 0.39 * decay_rate*Re
            
            
            
            #Animal carbon group                                                 # källor på detta möjligen via VERA
            
            c_from_animals_to_food = 0.40            # The amount of carbon that is pure food fron the animals
                                                    
                                                    
            carbon_respiration_factor_animals = 0.20        #amount of carbon that goes out to atm via respiration 
            
            
            carbon_manure_factor_from_animals = 0.40 
            

            #manure group
            
            manure_to_biogas_prodution_factor = 0.50          # the amount of manure that goes to the biogas chanber 
            
            manure_to_the_fields_as_manure = 0.50 

            manure_leaving_the_system_factor =  0      # amount of carbon from manure handeling and storage that is gone to the air 

          
             
            #Biogas group

            biogas_efficiency_Factor = 0.60     # this should be a number of the efficency of the biogas converstion from manure to gas. 
            
            biogas_production_from_manure_producing_biomanure = (1-biogas_efficiency_Factor)

            biogas_C_factor_used_inside_system = 0.50      # the amount of biogas used inside the system of the created biogas
            
            biogas_C_factor_leaving_system = (1-biogas_C_factor_used_inside_system)                 # the exported amount of biogas 
            
            #biochar group 
            
            biochar_efficiency_Factor = 0.50
            
            carbon_leaving_biochar_production = 1-biochar_efficiency_Factor
           
            
            #Ammendments grouped
            

            biogas_rest_product_to_bio_manure = 0.2             # assumtion of how much restproducts there are after  - biomanure
            
            
            catchment_crops_winter_factor = 0.1
            
            catchment_crops_spring_factor = 0.2
            
            carbon_effect_catchment_cropp = 300
            
            
            

            def Carbon_input_data_Based_on_HARVEST():
                #### the main input file ###

                ### The main input consits of harvest data - how much ton/ha is harvested per crop group - area - water content etc... ###

                df_input = pd.read_excel(r"")

                #print(df_input)

                global hostvete_in # för att ta in skördenivåerna för höstvete i medeltal för de senaste tre åren

                hostvete_fromfile = df_input.loc[df_input.crop =='wintercereal','3y_mean_harvest'].astype(int) # kg / ha of harvest for crop

                sgi_fromfile = df_input.loc[df_input.crop =='wintercereal','straw_ratio'].astype(float)       # straw:grain Ratio for crop

                hostvete_in = hostvete_fromfile

                sgi = sgi_fromfile

                # in drymatter of the crops, it is about 40 % carbon in total.
                c = 0.45        # 40 % carbon in dm
                dmfactor = df_input.loc[df_input.crop =='wintercereal','h20_'].astype(float)
                dm = (1-dmfactor)       # minus 14 % water content in crops

               
                HI = 0.40
            
                SR = 5.6  # specific for the winter wheat crop
                
                Bulk_harvest = hostvete_in*dm * c
                Bulk_straw = (hostvete_in*dm * c *(1-HI))/HI
                Bulk_roots = (hostvete_in*dm * c)/(SR * HI)     
                Bulk_root_extra = Bulk_roots* 0.65  #  the value is from the BOLINDER 2007 paper 
                
                Total_bulk_drymatter = (Bulk_harvest + Bulk_straw + Bulk_roots + Bulk_root_extra)
                
        
                ''' DOINT THE BOLINDER METHOD'''
                
                carbon_harvest = Total_bulk_drymatter * 0.322
                carbon_straw = Total_bulk_drymatter * 0.482
                carbon_rootz_1 = Total_bulk_drymatter  * 0.118
                carbon_rootz_2 = Total_bulk_drymatter  * 0.078
                carbon_roots = (carbon_rootz_1 + carbon_rootz_2)
                
                total_Carbon_bolinder = carbon_harvest + carbon_straw + carbon_roots
                
                # print(Total_bulk_drymatter *c)
                # print(Bulk_harvest *c)
                # print(Bulk_straw *c)
                # print(Bulk_roots *c + Bulk_root_extra *c)
                
                # print(total_Carbon_bolinder)
                
                print(carbon_harvest)
                print(carbon_straw)
                print(carbon_rootz_1)
                print(carbon_rootz_2)
                
                print(carbon_harvest)
                print(carbon_straw)
                print(carbon_rootz_1)
                print(carbon_rootz_2)
                

                
                
               
                def Definition_of_all_carbon_flows_from_harvest_and_straw():
                    

                    
                    
                    df_working_crop['AREAL']=1

                    df_working_crop['C_prod'] = df_working_crop.apply(lambda x : x['AREAL']*carbon_harvest, axis = 1)

                    df_working_crop['C_residue_total'] = df_working_crop.apply(lambda x : x['AREAL']*carbon_straw, axis = 1)

                    df_working_crop['C_root'] = df_working_crop.apply(lambda x : x['AREAL']*carbon_roots, axis = 1)
                    
                    df_working_crop['C_effect_catchmentcrops'] = df_working_crop.apply(lambda x : x['AREAL']*catchment_crops_winter_factor * carbon_effect_catchment_cropp, axis = 1)

                    df_working_crop['Total_biomass_carbon'] = df_working_crop.apply(lambda x : x['C_residue_total'] + x['C_prod'] + x['C_root'] +x['C_effect_catchmentcrops'], axis = 1)
                    
                    print(df_working_crop)

                    # Above is the core carbon content per hectar in the landscape. 
                    
                    # FIRST pathways 

                    df_working_crop['c_residue_barged'] = df_working_crop.apply(lambda x : x['C_residue_total']* barged_factor , axis = 1)

                    df_working_crop['C_residue_left'] = df_working_crop.apply(lambda x : x['C_residue_total'] *left_to_soil_factor, axis = 1)

                    df_working_crop['C_harvest_out'] = df_working_crop.apply(lambda x : x['C_prod'] *harvest_out_constant, axis = 1)
                    
                    df_working_crop['C_fodder_import'] = carbon_in_fodder_import
                    df_working_crop['C_A_New_manure_input'] = VERA_database_amount_of_carbon_in_manure_per_hectar_total
                    df_working_crop['CA_natural_added_manure_inpu'] = Calculated_pasture_added_carbon
                    
                    
                    # SECOND pathways 
                    
                    #fodder
                    df_working_crop['C_fodder_harv_imp'] = df_working_crop.apply(lambda x : x['C_prod']* fodder_constant_from_harvest  + x['C_fodder_import'], axis = 1) # här ingår det även lite halm eller restproduklt som något som kan gå till djuren
                    df_working_crop['C_residue_to_animal_fodder'] = df_working_crop.apply(lambda x :  x['c_residue_barged']*residue_carbon_to_animal_fodder, axis = 1) # Denna skall sen gå vidare till Manure total liksom 
                    df_working_crop['C_residue_to_stroe_animals'] = df_working_crop.apply(lambda x :  x['c_residue_barged']*residue_carbon_to_animal, axis = 1) # Denna skall sen gå vidare till Manure total liksom
                    df_working_crop['C_in_animal_box_total_fodder'] = df_working_crop.apply(lambda x :  x['C_fodder_harv_imp'] + x['C_residue_to_stroe_animals'] , axis = 1) # Denna skall sen gå vidare till Manure total liksom
                    #df_working_crop['C_in_animal_box_total_fodder'] = df_working_crop.apply(lambda x : ['C_total_fodder_box_to_animals'], axis = 1) # Denna skall sen gå vidare till Manure total liksom 
                    
                    
                    #food
                    df_working_crop['C_food_harv'] = df_working_crop.apply(lambda x : x['C_harvest_out']*c_food_out_human, axis = 1)
                    df_working_crop['c_export_out'] = df_working_crop.apply(lambda x : x['C_harvest_out']*c_export_out, axis = 1)
                    df_working_crop['c_industry_out'] = df_working_crop.apply(lambda x : x['C_harvest_out']*c_industry_out, axis = 1)
                    
                    df_working_crop['c_from_animals_to_food'] = df_working_crop.apply(lambda x : x['C_in_animal_box_total_fodder'] * c_from_animals_to_food, axis = 1)
                    df_working_crop['C_food_tot'] = df_working_crop.apply(lambda x : x['C_food_harv'] + x['c_from_animals_to_food'], axis = 1)

                    
                    
                    # THIRD pathways
  
                    #df_working_crop['C_A_manure'] = df_working_crop.apply(lambda x : x['C_in_animal_box_total_fodder']* carbon_manure_factor_from_animals, axis = 1)
                    
                    df_working_crop['C_A_manure'] = df_working_crop['C_A_New_manure_input']
                    
                    df_working_crop['C_A_Natural_added_manure'] = df_working_crop['CA_natural_added_manure_inpu']
                    
                   
                    
                    df_working_crop['carbon_respiration_factor_animals'] = df_working_crop.apply(lambda x : x['C_in_animal_box_total_fodder']* carbon_respiration_factor_animals, axis = 1)



                    # Scenario effect
                    
                    df_working_crop['C_residue_biogas'] = df_working_crop.apply(lambda x : x['c_residue_barged']*residues_biogas_factor, axis = 1)


                    df_working_crop['C_residue_bioCHAR'] = df_working_crop.apply(lambda x : x['c_residue_barged']*residues_bioCHAR_factor, axis = 1)




                    df_working_crop['C_roots_to_SOC_pool'] = df_working_crop.apply(lambda x : x['C_root'] *ICBM_total_factor_root, axis = 1)

                    df_working_crop['C_residue_to_SOC_pool'] = df_working_crop.apply(lambda x : x['C_residue_left']*ICBM_total_factor_straw, axis = 1)

                    
                    

                    #Manure 
                    df_working_crop['Manure_Biogas'] = df_working_crop.apply(lambda x : x['C_A_manure']* manure_to_biogas_prodution_factor, axis = 1)

                    df_working_crop['Manure_fields'] = df_working_crop.apply(lambda x : x['C_A_manure'] *manure_to_the_fields_as_manure, axis = 1)
                    df_working_crop['Manure_fields_already_There'] = df_working_crop['C_A_Natural_added_manure']
                    
                    df_working_crop['Total_manure_fields'] = df_working_crop['Manure_fields'] + df_working_crop['Manure_fields_already_There']
                    
                    df_working_crop['Manure_out_storage'] = df_working_crop.apply(lambda x : x['C_A_manure'] * manure_leaving_the_system_factor, axis = 1)
                    
                    
                    
                    #Biogas and BioChar 
                    
                    df_working_crop['Biogas_produced'] = df_working_crop.apply(lambda x : (x['Manure_Biogas'] + x['C_residue_biogas']) * biogas_efficiency_Factor, axis = 1)
                    
                    df_working_crop['Bio_manure_produced_by_biogas_production'] = df_working_crop.apply(lambda x : (x['Manure_Biogas'] + x['C_residue_biogas']) * biogas_production_from_manure_producing_biomanure , axis = 1)
            
                    #Above here is now 100% of carbon that went into the biogas chamber accounted for
                    
                    df_working_crop['Biochar_produced_carbon'] = df_working_crop.apply(lambda x : x['C_residue_bioCHAR'] * biochar_efficiency_Factor, axis = 1)
                    df_working_crop['Biochar_production_carbon_leaving_process'] = df_working_crop.apply(lambda x : x['C_residue_bioCHAR'] * carbon_leaving_biochar_production, axis = 1)
              
              
                    #Biogas use 
                    
                    df_working_crop['Biogas_to_use'] = df_working_crop.apply(lambda x : x['Biogas_produced'] * biogas_C_factor_used_inside_system , axis = 1)

                    df_working_crop['Biogas_out'] = df_working_crop.apply(lambda x : x['Biogas_produced'] *biogas_C_factor_leaving_system , axis = 1)

                
                    #Ammendments
                    
                    df_working_crop['biochar_stable_as_ammendment'] = df_working_crop.apply(lambda x : x['Biochar_produced_carbon'], axis = 1)
                    
                    df_working_crop['bio_manure_to_SOC'] = df_working_crop.apply(lambda x : x['Bio_manure_produced_by_biogas_production'], axis = 1)
                    
                    df_working_crop['Manure_amendment_to_SOC'] = df_working_crop.apply(lambda x : x['Total_manure_fields']*ICBM_total_factor_manure, axis = 1)
                    
                    df_working_crop['CC_extra_SOC'] = df_working_crop['C_effect_catchmentcrops']

                    
                    
                    print(df_working_crop)
                    print(df_working_crop.keys())
                    

                    #df_working_crop.to_excel(r'C:\Users\rehnan\OneDrive - Chalmers\Projektet\bridge_data_\csv_2022\paper_2\pkl\crop_dataframes_model/df_basefile_every_crop_testing_theory3.xlsx')
                    global df_swich_to_summation
                    df_swich_to_summation = pd.DataFrame(df_working_crop)

                    def summation_winterwheat_to_make_crop_df():
                        ''''''
                        global df_sum
                        df_sum = pd.DataFrame(df_swich_to_summation)
                  
                        def winterwheat():
                            
                            df_crop_ww_input = pd.DataFrame(df_sum)

                            crop_ww = df_crop_ww_input.T

                            df_summation_prell = crop_ww.copy()

                            df_summation_prell['Sum_1'] = df_summation_prell.sum(axis=1)

                            area_norm = df_crop_ww_input['AREAL'].sum()                  #Summering av hur stor areal det är av grödan - för att erhålla värden av kolflöden / hektar

                            df_crop_ww_prel = df_summation_prell.T # transpose the dataframe back - now with summation

                            global df1_sum

                            #print(df_crop_ww_prel)

                            #print(df_crop_ww_prel.keys())
                            
                            df1_sum1 = df_crop_ww_prel[['AREAL', 'C_prod',
                            'C_residue_total', 'C_root', 'C_effect_catchmentcrops',
                            'Total_biomass_carbon', 'c_residue_barged', 'C_residue_left',
                            'C_harvest_out', 'C_fodder_import', 'C_A_New_manure_input',
                            'CA_natural_added_manure_inpu', 'C_fodder_harv_imp',
                            'C_residue_to_animal_fodder', 'C_residue_to_stroe_animals',
                            'C_in_animal_box_total_fodder', 'C_food_harv', 'c_export_out',
                            'c_industry_out', 'c_from_animals_to_food', 'C_food_tot', 'C_A_manure',
                            'C_A_Natural_added_manure', 'carbon_respiration_factor_animals',
                            'C_residue_biogas', 'C_residue_bioCHAR', 'C_roots_to_SOC_pool',
                            'C_residue_to_SOC_pool', 'Manure_Biogas', 'Manure_fields',
                            'Manure_fields_already_There', 'Total_manure_fields',
                            'Manure_out_storage', 'Biogas_produced',
                            'Bio_manure_produced_by_biogas_production', 'Biochar_produced_carbon',
                            'Biochar_production_carbon_leaving_process', 'Biogas_to_use',
                            'Biogas_out', 'biochar_stable_as_ammendment', 'bio_manure_to_SOC',
                            'Manure_amendment_to_SOC', 'CC_extra_SOC']]
                            
                            df_sum_2 = df_crop_ww_prel[['AREAL','Total_biomass_carbon','C_roots_to_SOC_pool',
                            'C_residue_to_SOC_pool', 'biochar_stable_as_ammendment', 'bio_manure_to_SOC',
                            'Manure_amendment_to_SOC', 'CC_extra_SOC']]

                            df_sum_2 = df_sum_2.apply(lambda x: x/area_norm)
                            # Get the bottom row of the DataFrame
                            bottom_row = df_sum_2.iloc[-1]

                            # Select the columns you want to sum, excluding 'AREAL' and 'Total_biomass_carbon'
                            selected_columns = ['C_roots_to_SOC_pool', 'C_residue_to_SOC_pool', 'biochar_stable_as_ammendment',
                                                'bio_manure_to_SOC', 'Manure_amendment_to_SOC', 'CC_extra_SOC']

                            # Calculate the sum of the bottom row for the selected columns
                            #sum_bottom_row = bottom_row[selected_columns].sum()
                            df1_sum_winterwheat = df1_sum1.apply(lambda x: x/area_norm)

                            last_row_index = df1_sum_winterwheat.index[-1]

                            # Set the value in the 'AREAL' column of the last row
                            desired_value = 1  # Change this to the value you want to set
                            df1_sum_winterwheat.at[last_row_index, 'AREAL'] = desired_value
                            
                            
                            df1_sum_winterwheat.to_pickle(r'/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/paper_2/pkl/pkl_cropgroupes_2023/may_2023_version_ICBM_2/wintercereal.pkl')
                            
                            
                            print(bottom_rows_df)
                      
                            
                            
                        winterwheat()     
                    summation_winterwheat_to_make_crop_df()
                Definition_of_all_carbon_flows_from_harvest_and_straw()
            Carbon_input_data_Based_on_HARVEST()
        Scenario_factors_for_carbon_flow_analysis()
winterwheat_import()

print("fin")
