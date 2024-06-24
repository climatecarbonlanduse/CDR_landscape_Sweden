
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




''' BELOW VALUES ARE ONLY IF IT IS INCLUDED OR NOT AND FOR BIOCHAR _ THE SHARE OF THE TOTAL AVAIALBLE STRAW IN THE SCENARIO!'''
    
    
global max_biochar_effect
max_biochar_effect =   1     # 1 = All of the avaialbe straw that is availabe to actually barged extra is used to BIochar

global covercrops_choice
covercrops_choice =   1      # 1 = yes , 0 = no 



def structure_of_rotation_1():
    
    global df_wintercereals
    global df_springcereals
    global df_ley
    global df_legumes
    global df_other
    global df_rapeseed
    global df_fallow
    global df_ley_seed
    
    
    
    df_wintercereals = pd.read_pickle(r'')

    df_springcereals = pd.read_pickle(r'')

    df_ley = pd.read_pickle(r'')

    df_legumes = pd.read_pickle(r'')

    df_other = pd.read_pickle(r'')

    df_rapeseed = pd.read_pickle(r'')

    df_fallow = pd.read_pickle(r'')

    df_ley_seed = pd.read_pickle(r'')
    
    global df_ley_added
    
    df_ley_added = pd.read_pickle(r'')
   
    global df_covercrop
    df_covercrop = pd.read_pickle(r'')
   
    
    ''' follow the below examples '''
    
    
    def Rotation_R1_structure():
        
    

        df_list_R1 = []

        global df_wintercereals
        global df_springcereals
        global df_legumes
        global df_other
        global df_rapeseed
        global df_fallow
        global df_ley
        global df_covercrop
        

        # Original and the changed belwo
        #
        df_list_R1 =[df_wintercereals,df_springcereals,df_wintercereals,df_springcereals,df_springcereals,df_rapeseed]
        #df_list_R1 =[df_wintercereals,df_springcereals,df_ley,df_ley,df_springcereals,df_rapeseed,df_wintercereals,df_springcereals]

        #print(df_list)

        for i, crop in enumerate(df_list_R1):
            crop['Crop_nr'] = i + 1


        #print(df_list)

        # df_all = pd.DataFrame

        df_all = pd.concat([d[-1:] for d in df_list_R1], ignore_index=True)
        
        
        global rotation_first

        rotation_first = pd.DataFrame(df_all).copy()


        first_column = rotation_first.pop('Crop_nr')

        rotation_first.insert(0,'Crop_nr',first_column)
        rotation_first.set_index('Crop_nr')

        print("Below is the rotation first")
        print("This dataset is the bottom summation row of each crop within each rotation")
        print("FOR the Standard R1 rotation")
        print( rotation_first)


        #print("HÄR ÄR ÄR BRA")

    Rotation_R1_structure()
    
    def Rotation_R2_structure():


        df_list_R2 = []

        global df_wintercereals
        global df_springcereals
        global df_legumes
        global df_other
        global df_rapeseed
        global df_fallow4
        global df_ley

    
        df_list_R2 =[df_rapeseed,df_wintercereals,df_springcereals,df_wintercereals,df_legumes,df_wintercereals, df_springcereals, df_wintercereals] # 8 year
        #df_list_R2 =[df_rapeseed,df_wintercereals,df_springcereals,df_legumes,df_wintercereals,df_springcereals, df_springcereals, df_ley, df_ley] # 8 year

        #print(df_list)

        for i, crop in enumerate(df_list_R2):
            crop['Crop_nr'] = i + 1



        df_all = pd.concat([d[-1:] for d in df_list_R2], ignore_index=True)
        global rotation_second

        rotation_second = pd.DataFrame(df_all).copy()


        first_column = rotation_second.pop('Crop_nr')

        rotation_second.insert(0,'Crop_nr',first_column)
        rotation_second.set_index('Crop_nr')


    Rotation_R2_structure()
    
    def Rotation_R3_structure():


        df_list_R3 = []

        global df_wintercereals
        global df_springcereals
        global df_legumes
        global df_other
        global df_rapeseed
        global df_fallow
        global df_ley_seed

    
        df_list_R3 =[df_legumes,df_wintercereals,df_springcereals,df_ley_seed,df_ley_seed,df_wintercereals,df_springcereals,df_springcereals] #8 years 

        #print(df_list)

        for i, crop in enumerate(df_list_R3):
            crop['Crop_nr'] = i + 1


        #print(df_list)

        # df_all = pd.DataFrame

        df_all = pd.concat([d[-1:] for d in df_list_R3], ignore_index=True)
        global rotation_third

        rotation_third = pd.DataFrame(df_all).copy()


        first_column = rotation_third.pop('Crop_nr')

        rotation_third.insert(0,'Crop_nr',first_column)
        rotation_third.set_index('Crop_nr')


    Rotation_R3_structure()
    

    def Rotation_R4_structure():


        df_list_R4 = []

        global df_wintercereals
        global df_springcereals
        global df_legumes
        global df_other
        global df_rapeseed
        global df_fallow
        global df_ley


        df_list_R4 =[df_ley,df_ley,df_ley,df_springcereals,df_springcereals]

        #print(df_list)

        for i, crop in enumerate(df_list_R4):
            crop['Crop_nr'] = i + 1


        #print(df_list)

        # df_all = pd.DataFrame

        df_all = pd.concat([d[-1:] for d in df_list_R4], ignore_index=True)
        global rotation_fourth

        rotation_fourth = pd.DataFrame(df_all).copy()


        first_column = rotation_fourth.pop('Crop_nr')

        rotation_fourth.insert(0,'Crop_nr',first_column)
        rotation_fourth.set_index('Crop_nr')


    Rotation_R4_structure()
    
    def Rotation_R5_structure():


        df_list_R5 = []

        global df_wintercereals
        global df_springcereals
        global df_legumes
        global df_other
        global df_rapeseed
        global df_fallow
        global df_ley

    
        df_list_R5 =[df_ley,df_ley,df_ley,df_ley,df_wintercereals,df_springcereals, df_springcereals]

        #print(df_list)

        for i, crop in enumerate(df_list_R5):
            crop['Crop_nr'] = i + 1


        #print(df_list)

        # df_all = pd.DataFrame

        df_all = pd.concat([d[-1:] for d in df_list_R5], ignore_index=True)
        global rotation_fifth

        rotation_fifth = pd.DataFrame(df_all).copy()


        first_column = rotation_fifth.pop('Crop_nr')

        rotation_fifth.insert(0,'Crop_nr',first_column)
        rotation_fifth.set_index('Crop_nr')


    Rotation_R5_structure()
        
structure_of_rotation_1()  
