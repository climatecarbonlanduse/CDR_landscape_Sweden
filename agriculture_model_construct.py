


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
import fiona as fiona
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import os


#### Build on the previous scrips

def agricultural_system(): 
        
        def Standard_rotation_1():
            

            scenario_cereal_list = []

            ''' constructing a cereal rotation scheme for a scenario'''
            # This become  6 + 6 + 8 + 8 + 6 years = 34 years 
            scenario_cereal_list = [rotation_R1] # 32 år av crops

            df_all_summations = pd.concat(scenario_cereal_list)

            global df_scenario_sum

            df_scenario_sum = df_all_summations
            #print(df_scenario_sum)

            global df_summation_prell_all_rotation

            df_summation_prell_all_rotation_USE_AFTER_APPEND = df_scenario_sum

            df_summation_prell_all_rotation = df_scenario_sum.copy()
            
            df_summation_of_added_SOC = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']]

            df_summation_of_added_SOC['biochar_stable_as_ammendment'] = df_summation_of_added_SOC['biochar_stable_as_ammendment']*max_biochar_effect

            


            print(df_summation_of_added_SOC)
            global df_SUMMAN
            df_SUMMAN_T = df_summation_of_added_SOC.T
            
            df_SUMMAN = pd.DataFrame(df_summation_of_added_SOC).copy()
            df_SUMMAN['summation'] = 0
            df_SUMMAN['summation'] = df_SUMMAN_T.sum() #numeric_only='None', axis=1
            
            #print(df_SUMMAN)
            
            
            df_summation_prell_all_rotation['summation'] = 0

            df_summation_prell_all_rotation['summation'] = df_summation_prell_all_rotation.sum(axis=1) #numeric_only='None', axis=1

            df_summation_prell_all_rotation = pd.DataFrame(df_summation_prell_all_rotation)
            
        
            df_summation_prell_all_rotation_USE_AFTER_APPEND[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']] = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool', 'C_residue_to_SOC_pool']]

            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment']*max_biochar_effect
            
            
            
            #df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'] =  df_SUMMAN['summation']

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_straw'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_residue_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_roots'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_roots_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_manure'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['Manure_amendment_to_SOC'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_biochar'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'].values #
            
            global df_added_carbon_c_1

            df_added_carbon_c_1 = pd.DataFrame(df_summation_prell_all_rotation_USE_AFTER_APPEND[['Rot_nr','Crop_nr','Stable_carbon_straw','Stable_carbon_roots','Stable_carbon_manure','Stable_carbon_biochar']])
            df_added_carbon_c_1[['AREAL']] =  df_all_summations[['AREAL']]
            
            
            
            print("Checking Biochar_effect_1")
            print(df_summation_prell_all_rotation_USE_AFTER_APPEND)

            global added_carbon_list_working_dynamics
            added_carbon_list_working_dynamics = df_added_carbon_c_1
        
            
            print("THIS IS Standard rotation 1")
            print(df_added_carbon_c_1)

        Standard_rotation_1()
        
        
        
        def Standard_rotation_2():
            

            scenario_cereal_list = []

            ''' constructing a cereal rotation scheme for a scenario'''
            # This become  6 + 6 + 8 + 8 + 6 years = 34 years 
            scenario_cereal_list = [rotation_R2] # 32 år av crops

            df_all_summations = pd.concat(scenario_cereal_list)

            global df_scenario_sum

            df_scenario_sum = df_all_summations
            #print(df_scenario_sum)

            global df_summation_prell_all_rotation

            df_summation_prell_all_rotation_USE_AFTER_APPEND = df_scenario_sum

            df_summation_prell_all_rotation = df_scenario_sum.copy()
            
            df_summation_of_added_SOC = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']]

            
            df_summation_of_added_SOC['biochar_stable_as_ammendment'] = df_summation_of_added_SOC['biochar_stable_as_ammendment']*max_biochar_effect
            
            #print(df_summation_of_added_SOC)
            global df_SUMMAN
            df_SUMMAN_T = df_summation_of_added_SOC.T
            
            df_SUMMAN = pd.DataFrame(df_summation_of_added_SOC).copy()
            df_SUMMAN['summation'] = 0
            df_SUMMAN['summation'] = df_SUMMAN_T.sum() #numeric_only='None', axis=1
            
            #print(df_SUMMAN)
            
            
            df_summation_prell_all_rotation['summation'] = 0

            df_summation_prell_all_rotation['summation'] = df_summation_prell_all_rotation.sum(axis=1) #numeric_only='None', axis=1

            df_summation_prell_all_rotation = pd.DataFrame(df_summation_prell_all_rotation)
            
        
            df_summation_prell_all_rotation_USE_AFTER_APPEND[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']] = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool', 'C_residue_to_SOC_pool']]

            
            
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment']*max_biochar_effect
            
            
            
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'] =  df_SUMMAN['summation']

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_straw'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_residue_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_roots'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_roots_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_manure'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['Manure_amendment_to_SOC'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_biochar'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'].values #
            
            global df_added_carbon_c_2

            df_added_carbon_c_2 = pd.DataFrame(df_summation_prell_all_rotation_USE_AFTER_APPEND[['Rot_nr','Crop_nr','Stable_carbon_straw','Stable_carbon_roots','Stable_carbon_manure','Stable_carbon_biochar']])
            df_added_carbon_c_2[['AREAL']] =  df_all_summations[['AREAL']]

            global added_carbon_list_working_dynamics
            added_carbon_list_working_dynamics = df_added_carbon_c_2
        
            
            print("THIS IS Standard rotation 2")
            print(df_added_carbon_c_2)

        Standard_rotation_2()
        
        def Standard_rotation_3():
            

            scenario_cereal_list = []

            ''' constructing a cereal rotation scheme for a scenario'''
            # This become  6 + 6 + 8 + 8 + 6 years = 34 years 
            scenario_cereal_list = [rotation_R3] # 32 år av crops

            df_all_summations = pd.concat(scenario_cereal_list)

            global df_scenario_sum

            df_scenario_sum = df_all_summations
            #print(df_scenario_sum)

            global df_summation_prell_all_rotation

            df_summation_prell_all_rotation_USE_AFTER_APPEND = df_scenario_sum

            df_summation_prell_all_rotation = df_scenario_sum.copy()
            
            df_summation_of_added_SOC = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']]

            
            df_summation_of_added_SOC['biochar_stable_as_ammendment'] = df_summation_of_added_SOC['biochar_stable_as_ammendment']*max_biochar_effect
            
            #print(df_summation_of_added_SOC)
            global df_SUMMAN
            df_SUMMAN_T = df_summation_of_added_SOC.T
            
            df_SUMMAN = pd.DataFrame(df_summation_of_added_SOC).copy()
            df_SUMMAN['summation'] = 0
            df_SUMMAN['summation'] = df_SUMMAN_T.sum() #numeric_only='None', axis=1
            
            #print(df_SUMMAN)
            
            
            df_summation_prell_all_rotation['summation'] = 0

            df_summation_prell_all_rotation['summation'] = df_summation_prell_all_rotation.sum(axis=1) #numeric_only='None', axis=1

            df_summation_prell_all_rotation = pd.DataFrame(df_summation_prell_all_rotation)
            
        
            df_summation_prell_all_rotation_USE_AFTER_APPEND[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']] = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool', 'C_residue_to_SOC_pool']]

            
            
            
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment']*max_biochar_effect
            
            
            
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'] =  df_SUMMAN['summation']

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'].values #
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_straw'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_residue_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_roots'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_roots_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_manure'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['Manure_amendment_to_SOC'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_biochar'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'].values #
            
            global df_added_carbon_c_3

            df_added_carbon_c_3 = pd.DataFrame(df_summation_prell_all_rotation_USE_AFTER_APPEND[['Rot_nr','Crop_nr','Stable_carbon_straw','Stable_carbon_roots','Stable_carbon_manure','Stable_carbon_biochar']])
            df_added_carbon_c_3[['AREAL']] =  df_all_summations[['AREAL']]

            global added_carbon_list_working_dynamics
            added_carbon_list_working_dynamics = df_added_carbon_c_3
        
            
            
            print("THIS IS Standard rotation 3")
            print(df_added_carbon_c_3)

        Standard_rotation_3()
      
        
        def Standard_rotation_4():
            

            scenario_dairy_list = []

            ''' constructing a cereal rotation scheme for a scenario'''
            # This become  5 + 5 + 7 + 7 + 5 + 5 years = 34 years 
            scenario_dairy_list = [rotation_R4] # 34 år av crops

            df_all_summations = pd.concat(scenario_dairy_list)

            global df_scenario_sum

            df_scenario_sum = df_all_summations
            #print(df_scenario_sum)

            global df_summation_prell_all_rotation

            df_summation_prell_all_rotation_USE_AFTER_APPEND = df_scenario_sum

            df_summation_prell_all_rotation = df_scenario_sum.copy()
            
            df_summation_of_added_SOC = df_summation_prell_all_rotation[['CC_extra_SOC','biochar_stable_as_ammendment', 'Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']]

            
            df_summation_of_added_SOC['biochar_stable_as_ammendment'] = df_summation_of_added_SOC['biochar_stable_as_ammendment']*max_biochar_effect
            
            #print(df_summation_of_added_SOC)
            global df_SUMMAN
            df_SUMMAN_T = df_summation_of_added_SOC.T
            
            df_SUMMAN = pd.DataFrame(df_summation_of_added_SOC).copy()
            df_SUMMAN['summation'] = 0
            df_SUMMAN['summation'] = df_SUMMAN_T.sum() #numeric_only='None', axis=1
            
            #print(df_SUMMAN)
            
            
            df_summation_prell_all_rotation['summation'] = 0

            df_summation_prell_all_rotation['summation'] = df_summation_prell_all_rotation.sum(axis=1) #numeric_only='None', axis=1

            df_summation_prell_all_rotation = pd.DataFrame(df_summation_prell_all_rotation)
            
        
            df_summation_prell_all_rotation_USE_AFTER_APPEND[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']] = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool', 'C_residue_to_SOC_pool']]

            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment']*max_biochar_effect
            
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'] =  df_SUMMAN['summation']

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'].values #
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_straw'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_residue_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_roots'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_roots_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_manure'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['Manure_amendment_to_SOC'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_biochar'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'].values #
            
            global df_added_carbon_c4

            df_added_carbon_c4 = pd.DataFrame(df_summation_prell_all_rotation_USE_AFTER_APPEND[['Rot_nr','Crop_nr','Stable_carbon_straw','Stable_carbon_roots','Stable_carbon_manure','Stable_carbon_biochar']])
            global added_carbon_list_working_dynamics
            
            added_carbon_list_working_dynamics = df_added_carbon_c4
            df_added_carbon_c4[['AREAL']] =  df_all_summations[['AREAL']]
            
            print("THIS IS Standard rotation 4")
            print(df_added_carbon_c4)

        Standard_rotation_4()

        def Standard_rotation_5():
            

            scenario_dairy_list = []

            ''' constructing a cereal rotation scheme for a scenario'''
            # This become  5 + 5 + 7 + 7 + 5 + 5 years = 34 years 
            scenario_dairy_list = [rotation_R5] # 34 år av crops

            df_all_summations = pd.concat(scenario_dairy_list)

            global df_scenario_sum

            df_scenario_sum = df_all_summations
            #print(df_scenario_sum)

            global df_summation_prell_all_rotation

            df_summation_prell_all_rotation_USE_AFTER_APPEND = df_scenario_sum

            df_summation_prell_all_rotation = df_scenario_sum.copy()
            
            df_summation_of_added_SOC = df_summation_prell_all_rotation[['CC_extra_SOC','biochar_stable_as_ammendment', 'Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']]

            
            
            df_summation_of_added_SOC['biochar_stable_as_ammendment'] = df_summation_of_added_SOC['biochar_stable_as_ammendment']*max_biochar_effect
            
            #print(df_summation_of_added_SOC)
            global df_SUMMAN
            df_SUMMAN_T = df_summation_of_added_SOC.T
            
            df_SUMMAN = pd.DataFrame(df_summation_of_added_SOC).copy()
            df_SUMMAN['summation'] = 0
            df_SUMMAN['summation'] = df_SUMMAN_T.sum() #numeric_only='None', axis=1
            
            
            df_summation_prell_all_rotation['summation'] = 0

            df_summation_prell_all_rotation['summation'] = df_summation_prell_all_rotation.sum(axis=1) #numeric_only='None', axis=1

            df_summation_prell_all_rotation = pd.DataFrame(df_summation_prell_all_rotation)
            
        
            df_summation_prell_all_rotation_USE_AFTER_APPEND[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool','C_residue_to_SOC_pool']] = df_summation_prell_all_rotation[['CC_extra_SOC', 'biochar_stable_as_ammendment','Manure_amendment_to_SOC','C_roots_to_SOC_pool', 'C_residue_to_SOC_pool']]

            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment']*max_biochar_effect
            
            
            df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'] =  df_SUMMAN['summation']

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['summation'].values #
            
            global df_added_carbon_c5

            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_straw'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_residue_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_roots'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['C_roots_to_SOC_pool'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_manure'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['Manure_amendment_to_SOC'].values #
            df_summation_prell_all_rotation_USE_AFTER_APPEND['Stable_carbon_biochar'] = df_summation_prell_all_rotation_USE_AFTER_APPEND['biochar_stable_as_ammendment'].values #
            
            global df_added_carbon_c5

            df_added_carbon_c5 = pd.DataFrame(df_summation_prell_all_rotation_USE_AFTER_APPEND[['Rot_nr','Crop_nr','Stable_carbon_straw','Stable_carbon_roots','Stable_carbon_manure','Stable_carbon_biochar']])
            global added_carbon_list_working_dynamics
            added_carbon_list_working_dynamics = df_added_carbon_c5
            df_added_carbon_c5[['AREAL']] =  df_all_summations[['AREAL']]
            
            print("THIS IS Standard rotation 5")
            print(df_added_carbon_c5)

        Standard_rotation_5()
        
    agricultural_system()

    def Model_preperation_making_lists():
        
    
        ''' BELOW IS WHEN WE HAVE CONSIDERED THE NEW METHOD AS FOR AGRONIMICALLY ACCURATY CC SUGGESTIONS '''
        
        global Standard_rotaion_Area_1  
        Standard_rotaion_Area_1 = 4616
        global Standard_rotaion_Area_2
        Standard_rotaion_Area_2 = 5190
        global Standard_rotaion_Area_3
        Standard_rotaion_Area_3 =8320
        global Standard_rotaion_Area_4
        Standard_rotaion_Area_4 = 7990
        global Standard_rotaion_Area_5
        Standard_rotaion_Area_5 = 9408
        
        
        global standard_rotation_area_sr1
        standard_rotation_area_sr1 = 2308
        global standard_rotation_area_sr2 
        standard_rotation_area_sr2 = 1730
        
        global standard_rotation_area_sr3 
        standard_rotation_area_sr3= 0
        global check_area
        check_area = Standard_rotaion_Area_1 + Standard_rotaion_Area_2 + Standard_rotaion_Area_3  + Standard_rotaion_Area_4 + Standard_rotaion_Area_5 + standard_rotation_area_sr3 + standard_rotation_area_sr2 + standard_rotation_area_sr1
        
        
        print("THIS IS WHERE I CHECK AREA")
        print(check_area)
        global Standard_rotaion_Area_TOTAL
        Standard_rotaion_Area_TOTAL = 39562
        

        def Total_SR1():
            
            # Making a mean for each value since there are many fields in rotation at the same time

            total_C_1 = df_added_carbon_c_1

            global total_C_new_cereal_1
            total_C_new_cereal_1 = pd.DataFrame(total_C_1)

            print("THIS IS WHERE I MAKE THE CHANGE OF COVER CROPS for ROTATION 1 ADDING Rotation 1")

            total_C_new_cereal_1['Stable_CC'] = 0
            print(total_C_new_cereal_1)

            # Add 300 to the 'Stable_carbon' value in the first row
            total_C_new_cereal_1.loc[0, 'Stable_carbon_roots'] += (300 * covercrops_choice)

            print(total_C_new_cereal_1)
                        
            # Compute mean for each column
            mean_values = total_C_new_cereal_1.mean()

            # Compute sum for Stable_CC and Stable_carbon_biochar
            mean_values['Stable_CC'] = total_C_new_cereal_1['Stable_CC'].sum()
            mean_values['Stable_carbon_biochar'] = total_C_new_cereal_1['Stable_carbon_biochar'].mean()

            # Append the computed values as a new row
            total_C_new_cereal_1 = total_C_new_cereal_1.append(mean_values, ignore_index=True)

            print("Below is the total C STANDARD ROTATION 1 with mean values (and sum for Stable_CC and Stable_carbon_biochar)")
            print(total_C_new_cereal_1)


        
        Total_SR1()
        
        def Total_SR2():
            
            ''' MAKING A MEAN FOR EACH VALUE _ SINCE THERE ARE MANY MANY FIELDS IN TOTAL IN ROATION AT THE SAME TIME '''
            
            total_C_2 = df_added_carbon_c_2
            global total_C_new_cereal_2
            
            total_C_new_cereal_2 = pd.DataFrame(total_C_2)
            
            print(total_C_new_cereal_2)
            total_C_new_cereal_2['Stable_CC'] = 0
            
            print( " ADDING COVER CROPS ROTATION 2 " )
            # Add 300 to the 'Stable_carbon' value in the first row
           
            # Add 300 to the 'Stable_carbon' value in the first row
            total_C_new_cereal_2.loc[2, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            total_C_new_cereal_2.loc[3, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            
            print(total_C_new_cereal_2)
                        
            # Compute mean for each column
            mean_values = total_C_new_cereal_2.mean()

            # Compute sum for Stable_CC and Stable_carbon_biochar
            mean_values['Stable_CC'] = total_C_new_cereal_2['Stable_CC'].sum()
            mean_values['Stable_carbon_biochar'] = total_C_new_cereal_2['Stable_carbon_biochar'].mean()

            # Append the computed values as a new row
            total_C_new_cereal_2 = total_C_new_cereal_2.append(mean_values, ignore_index=True)

            print("Below is the total C STANDARD ROTATION 2 with mean values (and sum for Stable_CC and Stable_carbon_biochar)")
            print(total_C_new_cereal_2)
            
            
        Total_SR2()
        
        def Total_SR3():
            
            ''' MAKING A MEAN FOR EACH VALUE _ SINCE THERE ARE MANY MANY FIELDS IN TOTAL IN ROATION AT THE SAME TIME '''
            
            total_C_3 = df_added_carbon_c_3
            
            global total_C_new_cereal_3
            
            total_C_new_cereal_3 = pd.DataFrame(total_C_3)
            
            
            
            print(total_C_new_cereal_3)
            total_C_new_cereal_3['Stable_CC'] = 0
            
        
            
            print( " ADDING COVER CROPS ROTATION 3 " )
           
            total_C_new_cereal_3.loc[1, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            total_C_new_cereal_3.loc[5, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            total_C_new_cereal_3.loc[6, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            total_C_new_cereal_3.loc[7, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            
            
            print(total_C_new_cereal_3)
                        
            # Compute mean for each column
            mean_values = total_C_new_cereal_3.mean()

            # Compute sum for Stable_CC and Stable_carbon_biochar
            mean_values['Stable_CC'] = total_C_new_cereal_3['Stable_CC'].sum()
            mean_values['Stable_carbon_biochar'] = total_C_new_cereal_3['Stable_carbon_biochar'].mean()

            # Append the computed values as a new row
            total_C_new_cereal_3 = total_C_new_cereal_3.append(mean_values, ignore_index=True)

            print("Below is the total C STANDARD ROTATION 3 with mean values (and sum for Stable_CC and Stable_carbon_biochar)")
            print(total_C_new_cereal_3)
              
            
        
        Total_SR3()
        
     
         
        def Total_SR4():
            
            ''' MAKING A MEAN FOR EACH VALUE _ SINCE THERE ARE MANY MANY FIELDS IN TOTAL IN ROATION AT THE SAME TIME '''
            
            total_C_4 = df_added_carbon_c4
            
            global total_C_new_cereal_4
            
            total_C_new_cereal_4 = pd.DataFrame(total_C_4)
            
            print(total_C_new_cereal_4)
            total_C_new_cereal_4['Stable_CC'] = 0
            
            
            # Compute mean for each column
            mean_values = total_C_new_cereal_4.mean()

            # Compute sum for Stable_CC and Stable_carbon_biochar
            mean_values['Stable_CC'] = total_C_new_cereal_4['Stable_CC'].sum()
            mean_values['Stable_carbon_biochar'] = total_C_new_cereal_4['Stable_carbon_biochar'].mean()

            # Append the computed values as a new row
            total_C_new_cereal_4 = total_C_new_cereal_4.append(mean_values, ignore_index=True)

            print("Below is the total C STANDARD ROTATION 4 with mean values (and sum for Stable_CC and Stable_carbon_biochar)")
            print(total_C_new_cereal_4)
            
            
        
        Total_SR4()
        
        
        
        def Total_SR5():
            
            ''' MAKING A MEAN FOR EACH VALUE _ SINCE THERE ARE MANY MANY FIELDS IN TOTAL IN ROATION AT THE SAME TIME '''
            
            total_C_5 = df_added_carbon_c5
            
            global total_C_new_cereal_5
            
            total_C_new_cereal_5 = pd.DataFrame(total_C_5)
            
            total_C_new_cereal_5['Stable_CC'] = 0
            
            print( " ADDING COVER CROPS ROTATION 5 " )
            # Add 300 to the 'Stable_carbon' value in the first row
            
            
            total_C_new_cereal_5.loc[4, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            total_C_new_cereal_5.loc[5, 'Stable_carbon_roots'] += (300 * covercrops_choice)
            
            print(total_C_new_cereal_5)
                        
            # Compute mean for each column
            mean_values = total_C_new_cereal_5.mean()

            # Compute sum for Stable_CC and Stable_carbon_biochar
            mean_values['Stable_CC'] = total_C_new_cereal_5['Stable_CC'].sum()
            mean_values['Stable_carbon_biochar'] = total_C_new_cereal_5['Stable_carbon_biochar'].mean()

            # Append the computed values as a new row
            total_C_new_cereal_5 = total_C_new_cereal_5.append(mean_values, ignore_index=True)

            print("Below is the total C STANDARD ROTATION 5 with mean values (and sum for Stable_CC and Stable_carbon_biochar)")
            print(total_C_new_cereal_5)
            
        
        
        Total_SR5()
    
        global timeframe

        timeframe = 30
        
        
        
        # Get last row of cereal
        last_row_c1 = total_C_new_cereal_1.iloc[-1]
        last_row_c2 = total_C_new_cereal_2.iloc[-1]
        last_row_c3 = total_C_new_cereal_3.iloc[-1]
        last_row_c4 = total_C_new_cereal_4.iloc[-1]
        last_row_c5 = total_C_new_cereal_5.iloc[-1]

        
        print("BELOW IS THE STANDARD FORAMTION ONE INFORMATION ")
        global new_df_cereal1
        new_df_cereal1 = pd.DataFrame({'Year': range(2023, 2053), 'Stable_carbon_straw': [last_row_c1['Stable_carbon_straw']] * timeframe,
                                       
                                       'Stable_carbon_roots': [last_row_c1['Stable_carbon_roots']] * timeframe,
                                       'Stable_carbon_manure': [last_row_c1['Stable_carbon_manure']] * timeframe,
                                       'Stable_carbon_biochar': [last_row_c1['Stable_carbon_biochar']] * timeframe, 
                                       'Stable_CC': [last_row_c1['Stable_CC']] * timeframe, 
                                       'AREAL': [last_row_c1['AREAL']] * timeframe })
        
        new_df_cereal1['results'] = 0
        
        print(new_df_cereal1)
        global new_df_cereal1_2
        new_df_cereal1_2 = new_df_cereal1.iloc[-1:, :4].reset_index(drop=True).astype(float)
        print(new_df_cereal1_2)

        
        
        
        
        print("BELOW IS THE STANDARD FORAMTION TWO INFORMATION ")
        global new_df_cereal2
        new_df_cereal2 = pd.DataFrame({'Year': range(2023, 2053), 'Stable_carbon_straw': [last_row_c2['Stable_carbon_straw']] * timeframe,
                                       
                                       'Stable_carbon_roots': [last_row_c2['Stable_carbon_roots']] * timeframe,
                                       'Stable_carbon_manure': [last_row_c2['Stable_carbon_manure']] * timeframe,
                                       'Stable_carbon_biochar': [last_row_c2['Stable_carbon_biochar']] * timeframe, 
                                       'Stable_CC': [last_row_c2['Stable_CC']] * timeframe, 
                                       'AREAL': [last_row_c2['AREAL']] * timeframe })
        
        new_df_cereal2['results'] = 0
        
        print(new_df_cereal2)
        global total_c_list_cereal2
        total_c_list_cereal2 = new_df_cereal2.values.tolist()
        
        new_df_cereal2_2 = new_df_cereal2.iloc[-1:, :4].reset_index(drop=True).astype(float)
        print(new_df_cereal2_2)
        
        
        print("BELOW IS THE STANDARD FORAMTION THREE INFORMATION ")
        global new_df_cereal3
        new_df_cereal3 = pd.DataFrame({'Year': range(2023, 2053), 'Stable_carbon_straw': [last_row_c3['Stable_carbon_straw']] * timeframe,
                                       
                                       'Stable_carbon_roots': [last_row_c3['Stable_carbon_roots']] * timeframe,
                                       'Stable_carbon_manure': [last_row_c3['Stable_carbon_manure']] * timeframe,
                                       'Stable_carbon_biochar': [last_row_c3['Stable_carbon_biochar']] * timeframe, 
                                       'Stable_CC': [last_row_c3['Stable_CC']] * timeframe, 
                                       'AREAL': [last_row_c3['AREAL']] * timeframe })
        
        new_df_cereal3['results'] = 0
        
        print(new_df_cereal3)
        global total_c_list_cereal3
        total_c_list_cereal3 = new_df_cereal3.values.tolist()
        
        new_df_cereal3_2 = new_df_cereal3.iloc[-1:, :4].reset_index(drop=True).astype(float)
        print(new_df_cereal3_2)
        
        
        
        print("BELOW IS THE STANDARD FORAMTION THREE INFORMATION ")
        global new_df_cereal4
        new_df_cereal4 = pd.DataFrame({'Year': range(2023, 2053), 'Stable_carbon_straw': [last_row_c4['Stable_carbon_straw']] * timeframe,
                                       
                                       'Stable_carbon_roots': [last_row_c4['Stable_carbon_roots']] * timeframe,
                                       'Stable_carbon_manure': [last_row_c4['Stable_carbon_manure']] * timeframe,
                                       'Stable_carbon_biochar': [last_row_c4['Stable_carbon_biochar']] * timeframe, 
                                       'Stable_CC': [last_row_c4['Stable_CC']] * timeframe, 
                                       'AREAL': [last_row_c4['AREAL']] * timeframe })
        
        new_df_cereal4['results'] = 0
        
        print(new_df_cereal4)
        global total_c_list_cereal4
        total_c_list_cereal4 = new_df_cereal4.values.tolist()
        
        new_df_cereal4_2 = new_df_cereal4.iloc[-1:, :4].reset_index(drop=True).astype(float)
        print(new_df_cereal4_2)
        
        
        
        print("BELOW IS THE STANDARD FORAMTION THREE INFORMATION ")
        global new_df_cereal5
        new_df_cereal5 = pd.DataFrame({'Year': range(2023, 2053), 'Stable_carbon_straw': [last_row_c5['Stable_carbon_straw']] * timeframe,
                                       
                                       'Stable_carbon_roots': [last_row_c5['Stable_carbon_roots']] * timeframe,
                                       'Stable_carbon_manure': [last_row_c5['Stable_carbon_manure']] * timeframe,
                                       'Stable_carbon_biochar': [last_row_c5['Stable_carbon_biochar']] * timeframe, 
                                       'Stable_CC': [last_row_c5['Stable_CC']] * timeframe, 
                                       'AREAL': [last_row_c5['AREAL']] * timeframe })
        
        new_df_cereal5['results'] = 0
        
        print(new_df_cereal5)
        global total_c_list_cereal5
        total_c_list_cereal5 = new_df_cereal5.values.tolist()
        
        new_df_cereal5_2 = new_df_cereal5.iloc[-1:, :4].reset_index(drop=True).astype(float)
        print(new_df_cereal5_2)
    
    Model_preperation_making_lists()   
