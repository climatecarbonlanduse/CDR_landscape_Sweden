
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


# Builds on previous scrips and subsequently previous imports of dataframes

''' Imports '''

def the_model_mechanics_and_functions_of_model(): 
        
        
        def previous_tests(): 
            
            def ICBM_Model_first_try(): 
                import numpy as np
                import pandas as pd
                from scipy.integrate import odeint
                import matplotlib.pyplot as plt
                
                
                ''' Defining the size of the carbon stock pools - total - the mix of both the existing Y and the O'''
                
                SOC_init_cereal = 67700
                SOC_init_cereal2 = 67700
                SOC_init_cereal3 = 69000 
                SOC_init_dairy = 75400
                SOC_init_dairy2 = 75400
                
            
                        # Parameters
                params = {
                    'ky': 0.758,
                    'ko': 0.00605,
                    'hm': 0.32,
                    'hs': 0.2,
                    'hr': 0.39
                }
                
                
            
                # Initial values for your carbon pools
                initial_C = [0, SOC_init_cereal]  # [Young, Old]

                def icbm_model(C, t, carbon_input_series, params):
                    carbon_input = carbon_input_series[int(t)] # Select the appropriate carbon input for this time step
                    Cy, Co = C
                    dCy_dt = carbon_input - (params['ky'] + params['hr']) * Cy + params['hm'] * Cy
                    
                    
                    dCo_dt = params['hs'] * Cy - params['ko'] * Co
                    return np.array([dCy_dt, dCo_dt], dtype=np.float64)


                initial_C_float64 = np.array(initial_C, dtype=np.float64)


                # Fetch the last row from the dataframe
                last_row = new_df_cereal1_2.iloc[-1]
                
                
                print("Kollar vilka värden det egentligen är - odeint krånglar")
                print(new_df_cereal1)
                
                
                
            
                # Run the model for each carbon input
                outputs = []
                carbon_columns = ['Stable_carbon_straw', 'Stable_carbon_roots', 'Stable_carbon_manure']

                # Assuming you've already defined the last_row properly
                for col in carbon_columns:
                    # Use the value from the last row of the respective column as the carbon input for the entire model run
                    
                    
                    carbon_input_value = float(last_row[col])  # Ensure this is a float
                    t = np.arange(10)
                    carbon_input_series = np.full(t.shape, carbon_input_value, dtype=np.float64)  # Create a float64 numpy array filled with the carbon input value

                    # Ensure the initial_C is a float64 numpy array
                    initial_C_float64 = np.array(initial_C, dtype=np.float64)

                    C = odeint(icbm_model, initial_C_float64, t, args=(carbon_input_series, params))

                    outputs.append(C)
                    
                    # Printing for debugging purposes
                    print("Column:", col)
                    print("Carbon input series:", carbon_input_series)
                    print("Initial C:", initial_C_float64)


                # Create a new dataframe for results
                R1_model_ICBM = pd.DataFrame({
                    'Straw': outputs[0][:, 0],
                    'Roots': outputs[1][:, 0],
                    'Manure': outputs[2][:, 0],
                    'Old': outputs[0][:, 1],
                    
                })
                
                
                print(R1_model_ICBM)
                
                
                # Plotting
                plt.figure()
                plt.plot(R1_model_ICBM.index, R1_model_ICBM['Straw'], label='Straw')
                plt.plot(R1_model_ICBM.index, R1_model_ICBM['Roots'], label='Roots')
                plt.plot(R1_model_ICBM.index, R1_model_ICBM['Manure'], label='Manure')
                plt.plot(R1_model_ICBM.index, R1_model_ICBM['Old'], label='Old')
            
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Winter Wheat Field')
                plt.savefig('ICBM_test.png', dpi=300)
                #plt.show()
            #ICBM_Model_first_try()

            def ICBM_SECOND_TRY():
        
        
                import pandas as pd
                import matplotlib.pyplot as plt
                
                print("Starting second ICBM try")

                global old_pool_share
                old_pool_share = 1

                # ICBM Model setup
                class ICBM:
                    def __init__(self, params):
                        self.ky = params['ky']
                        self.ko = params['ko']
                        self.hm = params['hm']
                        self.hs = params['hs']
                        self.hr = params['hr']

                    def run(self, SOC_init_cereal, Straw, Roots, Manure, years):
                        Y = SOC_init_cereal * (1-old_pool_share)
                        O = SOC_init_cereal * old_pool_share

                        SOC = [SOC_init_cereal]
                        Y_vals = [Y]
                        O_vals = [O]

                        for i in range(years):
                            delta_Y = (-self.ky * Y) + (self.hs * Straw) + (self.hr * Roots) + (self.hm * Manure)
                            delta_O = (-self.ko * O) + (1 - self.hs) * Straw + (1 - self.hr) * Roots + (1 - self.hm) * Manure

                            Y += delta_Y
                            O += delta_O

                            SOC.append(Y + O)
                            Y_vals.append(Y)
                            O_vals.append(O)

                        df = pd.DataFrame({
                            'Year': list(range(years + 1)),
                            'Total_SOC': SOC,
                            'Y_pool': Y_vals,
                            'O_pool': O_vals
                        })

                        return df

                # Parameters
                params = {
                    'ky': 0.758,
                    'ko': 0.00605,
                    'hm': 0.32,
                    'hs': 0.2,
                    'hr': 0.39
                }

                # Inputs
                SOC_init_cereal = 67700
                Straw = 459
                Roots = 517
                Manure = 48

                model = ICBM(params)
                df = model.run(SOC_init_cereal, Straw, Roots, Manure, 300)

                # Plotting
                plt.figure(figsize=(10,6))
                plt.plot(df['Year'], df['Total_SOC'], label='Total SOC')
                plt.plot(df['Year'], df['Y_pool'], label='Active (Y) Pool')
                plt.plot(df['Year'], df['O_pool'], label='Slow (O) Pool')
                plt.xlabel('Years')
                plt.ylabel('SOC (kg C ha^-1)')
                plt.title('ICBM Model Over 30 Years')
                plt.legend()
                plt.grid(True)
                #

                print(df)
            #ICBM_SECOND_TRY()
        #previous_tests()
            
            
        global SOC_init_cereal
        SOC_init_cereal = 67700
        global SOC_init_cereal2
        SOC_init_cereal2 = 67700
        global SOC_init_cereal3
        SOC_init_cereal3 = 69000 
        global SOC_init_dairy
        SOC_init_dairy = 75400
        global SOC_init_dairy3
        SOC_init_dairy2 = 75400
        
           
            
        ''' ROTATION 1 with all the inputs '''
        
        
        print("********************")
        print("********************")
        print("********************")
        print("********************")
        print("HERE IS EVERYTHING REGARDING ROTATION 1")
        print("HERE IS EVERYTHING REGARDING ROTATION 1")
        print("********************")
        print("********************")
            
        def ICBM_third_and_Final_good_model():
            
            R1_import = pd.DataFrame(new_df_cereal1)
            
            print("TITTAR PÅ R1 här innan assignment ")
            print(R1_import)

            
            def importing_added_rotation_carbon():
                ''''''
                
                global straw_R1
                straw_R1 = R1_import.iloc[0, 1]  # iloc[0, 1] refers to the first row and second column.

                
                global roots_R1
                roots_R1 = R1_import.iloc[0, 2]
                
                global manure_R1
                manure_R1 = R1_import.iloc[0, 3]
                
                global stable_CC_R1
                stable_CC_R1 = R1_import.iloc[0, 5]
                
                global biochar_R1
                biochar_R1 = R1_import.iloc[0, 4] 
                
                print(straw_R1)
                print(roots_R1)
                print(manure_R1)
                
                
                
            importing_added_rotation_carbon()
            
            
            
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib.pyplot as plt

            class model():
                def __init__(self, para, ini):
                    self.ky = para['ky']    
                    self.ko = para['ko']    
                    self.hm = para['hm']    
                    self.hs = para['hs']
                    self.hr = para['hr']
                    
                    self.Y_manure = ini['Y_manure']
                    self.Y_straw = ini['Y_straw']
                    self.Y_roots = ini['Y_roots']
                    self.O = ini['O']

                def compute(self, t, I_manure=0.0, I_straw=0.0, I_roots=0.0, Re=0.8):
                    
                    t = np.array(t, ndmin=1)
                    I_manure = I_manure*np.ones(np.shape(t))
                    I_straw = I_straw*np.ones(np.shape(t))
                    I_roots = I_roots*np.ones(np.shape(t))
                    Re = Re*np.ones(np.shape(t))
                    
                    nsteps = len(t)
                    C = np.zeros((4, nsteps))
                    C[0,0] = self.Y_manure
                    C[1,0] = self.Y_straw
                    C[2,0] = self.Y_roots
                    C[3,0] = self.O

                    for m in range(1,nsteps):
                        k = [self.ky * Re[m], self.ko * Re[m]]
                        h = [self.hm, self.hs, self.hr]
                        x0 = [self.Y_manure, self.Y_straw, self.Y_roots, self.O]
                        I = [I_manure[m], I_straw[m], I_roots[m]]
                        tspan = [t[m-1], t[m]]
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        for i in range(4):
                            C[i,m] = x[1,i]
                        self.Y_manure = x[1,0]
                        self.Y_straw = x[1,1]
                        self.Y_roots = x[1,2]
                        self.O = x[1,3]
                        
                    transfer_rates = []  # List to store the transfer rates from Y_straw to O at each time step
                    
                    for m in range(1,nsteps):
                        ...
                        # Check if the current year (or time step) is a 6th year
                        current_year = t[m]
                        if current_year % 6 == 0:
                            I_roots[m] += stable_CC_R1  # Add the Stable_CC to the Y_roots input
                        
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        transfer_rate = h[1] * k[0] * x[1, 1]  # h[1]*k[0]*Y_straw
                        transfer_rates.append(transfer_rate)
                        ...

                                                            
                    
                    print("Size of Y_straw at each time step:", C[1])
                    print("Transfer rate from Y_straw to O at each time step:", transfer_rates)
                        
                        
                        
                    return C

            def time_derivative(x, t, I, k, h):
                dy_manure_dt = I[0] - k[0]*x[0]
                dy_straw_dt = I[1] - k[0]*x[1]
                dy_roots_dt = I[2] - k[0]*x[2]
                do_dt = h[0]*k[0]*x[0] + h[1]*k[0]*x[1] + h[2]*k[0]*x[2] - k[1]*x[3]
                return [dy_manure_dt, dy_straw_dt, dy_roots_dt, do_dt]
            
            def test():
                # test ICBM for winter wheat field
                para = {
                        'ky': 0.758,     # yr-1
                        'ko': 0.00605,   # yr-1
                        'hm': 0.32,      # -
                        'hs': 0.2,       # -
                        'hr': 0.39       # -
                        }
                            
                ini = {
                    'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                    'Y_straw': straw_R1,
                    'Y_roots': roots_R1,
                    'O': SOC_init_cereal 
                    }


                
                added_C = {
                        'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                        'Y_straw': straw_R1,
                        'Y_roots': roots_R1
                        }
                            
                # create model instance
                run1 = model(para, ini)
                
                n = 30  # yr
                t = np.linspace(0, n)
                
                # run model and return results
                C = run1.compute(t, I_straw=added_C['Y_straw'],  I_roots=added_C['Y_roots'], Re=0.8)
                
                
                
                
                total_initial = np.sum(list(ini.values()))
                total_final = np.sum(C[:, -1])
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100
                
                print(f"Total accumulated added straw over 30 years: {increase_amount:.2f}")
                print(f"Amount transferred from the young straw pool to the old pool: {increase_percent:.2f}")

                # plot figure
                plt.figure()

                #plt.plot(t, C[0], label='Manure')
                plt.plot(t, C[1], label='Straw')
                plt.plot(t, C[2], label='Roots')

                plt.plot(t, C[3], label='Old')
                plt.plot(t, np.sum(C, axis=0), label='Total')
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Standrad rotation R1')
                plt.savefig('ICBM_test.png', dpi=300)
                
                #plt.show()
                
                # Calculate increase in the total pool after 30 years
                total_initial = np.sum(list(ini.values()))

                total_final = np.sum(C, axis=0)[-1]
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100

                print(f"Increase in total pool after 30 years: {increase_amount:.2f}")
                print(f"Percentage increase: {increase_percent:.2f}%")
                
                # Extract final values after 30 years
                final_manure = C[0, -1]
                final_straw = C[1, -1]
                final_roots = C[2, -1]
                final_old_pool = C[3, -1]
                final_total = np.sum(C[:, -1])

                print(f"Manure pool after 30 years: {final_manure:.2f}")
                print(f"Straw pool after 30 years: {final_straw:.2f}")
                print(f"Roots pool after 30 years: {final_roots:.2f}")
                print(f"Old pool after 30 years: {final_old_pool:.2f}")
                print(f"Total after 30 years: {final_total:.2f}")
                
                
                # Creating a DataFrame to store the yearly values
                data = {
                    'Year': t,
                    'Manure': C[0],
                    'Straw': C[1],
                    'Roots': C[2],
                    'Old': C[3],
                    'Total_ICBM': np.sum(C, axis=0)
                }
                
                global df_R1
                df_R1 = pd.DataFrame(data)
                return df_R1
                            
                

            test()
            
    
            
            df_R1['Stable_Biochar'] = biochar_R1

            df_R1['Cumulative_Biochar'] = df_R1['Stable_Biochar'].cumsum()
            df_R1['Total_Carbon'] = df_R1['Total_ICBM'] + df_R1['Cumulative_Biochar']


            
            
            
                      
            # Assuming df_R5 has 30 rows
            df_R1['Year'] = range(0, 50)

            # Setting 'Year' column as the index
            df_R1.set_index('Year', inplace=True)
            
            
            df_R1_save = pd.DataFrame(df_R1)

            df_R1_save = df_R1_save.iloc[:30]
            
            

                
            print(df_R1_save)

        ICBM_third_and_Final_good_model()
        
        
        ''' ROTATION 2 with all the inputs '''
        
        
        print("********************")
        print("********************")
        print("********************")
        print("********************")
        print("HERE IS EVERYTHING REGARDING ROTATION 2")
        print("HERE IS EVERYTHING REGARDING ROTATION 2")
        print("********************")
        print("********************")
            
        def ICBM_third_and_Final_good_model():
            
            R1_import = pd.DataFrame(new_df_cereal2)
            
            print("TITTAR PÅ R1 här innan assignment ")
            print(R1_import)

            
            def importing_added_rotation_carbon():
                ''''''
                
                global straw_R1
                straw_R1 = R1_import.iloc[0, 1]  # iloc[0, 1] refers to the first row and second column.

                
                global roots_R1
                roots_R1 = R1_import.iloc[0, 2]
                
                global manure_R1
                manure_R1 = R1_import.iloc[0, 3]
                
                global stable_CC_R1
                stable_CC_R1 = R1_import.iloc[0, 5]
                
                global biochar_R1
                biochar_R1 = R1_import.iloc[0, 4] 
                
                print(straw_R1)
                print(roots_R1)
                print(manure_R1)
                
                
                
            importing_added_rotation_carbon()
            
            
            
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib.pyplot as plt

            class model():
                def __init__(self, para, ini):
                    self.ky = para['ky']    
                    self.ko = para['ko']    
                    self.hm = para['hm']    
                    self.hs = para['hs']
                    self.hr = para['hr']
                    
                    self.Y_manure = ini['Y_manure']
                    self.Y_straw = ini['Y_straw']
                    self.Y_roots = ini['Y_roots']
                    self.O = ini['O']

                def compute(self, t, I_manure=0.0, I_straw=0.0, I_roots=0.0, Re=1):
                    
                    t = np.array(t, ndmin=1)
                    I_manure = I_manure*np.ones(np.shape(t))
                    I_straw = I_straw*np.ones(np.shape(t))
                    I_roots = I_roots*np.ones(np.shape(t))
                    Re = Re*np.ones(np.shape(t))
                    
                    nsteps = len(t)
                    C = np.zeros((4, nsteps))
                    C[0,0] = self.Y_manure
                    C[1,0] = self.Y_straw
                    C[2,0] = self.Y_roots
                    C[3,0] = self.O

                    for m in range(1,nsteps):
                        k = [self.ky * Re[m], self.ko * Re[m]]
                        h = [self.hm, self.hs, self.hr]
                        x0 = [self.Y_manure, self.Y_straw, self.Y_roots, self.O]
                        I = [I_manure[m], I_straw[m], I_roots[m]]
                        tspan = [t[m-1], t[m]]
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        for i in range(4):
                            C[i,m] = x[1,i]
                        self.Y_manure = x[1,0]
                        self.Y_straw = x[1,1]
                        self.Y_roots = x[1,2]
                        self.O = x[1,3]
                        
                    transfer_rates = []  # List to store the transfer rates from Y_straw to O at each time step
                    
                    for m in range(1,nsteps):
                        ...
                        # Check if the current year (or time step) is a 6th year
                        current_year = t[m]
                        if current_year % 6 == 0:
                            I_roots[m] += stable_CC_R1  # Add the Stable_CC to the Y_roots input
                        
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        transfer_rate = h[1] * k[0] * x[1, 1]  # h[1]*k[0]*Y_straw
                        transfer_rates.append(transfer_rate)
                        ...

                                                            
                    
                    print("Size of Y_straw at each time step:", C[1])
                    print("Transfer rate from Y_straw to O at each time step:", transfer_rates)
                        
                        
                        
                    return C

            def time_derivative(x, t, I, k, h):
                dy_manure_dt = I[0] - k[0]*x[0]
                dy_straw_dt = I[1] - k[0]*x[1]
                dy_roots_dt = I[2] - k[0]*x[2]
                do_dt = h[0]*k[0]*x[0] + h[1]*k[0]*x[1] + h[2]*k[0]*x[2] - k[1]*x[3]
                return [dy_manure_dt, dy_straw_dt, dy_roots_dt, do_dt]
            
            def test():
                # test ICBM for winter wheat field
                para = {
                        'ky': 0.758,     # yr-1
                        'ko': 6.05e-3,   # yr-1
                        'hm': 0.32,      # -
                        'hs': 0.2,       # -
                        'hr': 0.39       # -
                        }
                            
                ini = {
                    'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                    'Y_straw': straw_R1,
                    'Y_roots': roots_R1,
                    'O': SOC_init_cereal2 
                    }


                
                added_C = {
                        'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                        'Y_straw': straw_R1,
                        'Y_roots': roots_R1
                        }
                            
                # create model instance
                run1 = model(para, ini)
                
                n = 30  # yr
                t = np.linspace(0, n)
                
                # run model and return results
                C = run1.compute(t, I_straw=added_C['Y_straw'],  I_roots=added_C['Y_roots'], Re=0.8)
                
                
                
                
                total_initial = np.sum(list(ini.values()))
                total_final = np.sum(C[:, -1])
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100
                
                print(f"Total accumulated added straw over 30 years: {increase_amount:.2f}")
                print(f"Amount transferred from the young straw pool to the old pool: {increase_percent:.2f}")

                # plot figure
                plt.figure()

                #plt.plot(t, C[0], label='Manure')
                plt.plot(t, C[1], label='Straw')
                plt.plot(t, C[2], label='Roots')

                plt.plot(t, C[3], label='Old')
                plt.plot(t, np.sum(C, axis=0), label='Total')
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Standrad rotation R2')
                plt.savefig('ICBM_test.png', dpi=300)
                
                #plt.show()
                
                # Calculate increase in the total pool after 30 years
                total_initial = np.sum(list(ini.values()))

                total_final = np.sum(C, axis=0)[-1]
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100

                print(f"Increase in total pool after 30 years: {increase_amount:.2f}")
                print(f"Percentage increase: {increase_percent:.2f}%")
                
                # Extract final values after 30 years
                final_manure = C[0, -1]
                final_straw = C[1, -1]
                final_roots = C[2, -1]
                final_old_pool = C[3, -1]
                final_total = np.sum(C[:, -1])

                print(f"Manure pool after 30 years: {final_manure:.2f}")
                print(f"Straw pool after 30 years: {final_straw:.2f}")
                print(f"Roots pool after 30 years: {final_roots:.2f}")
                print(f"Old pool after 30 years: {final_old_pool:.2f}")
                print(f"Total after 30 years: {final_total:.2f}")
                
                
                # Creating a DataFrame to store the yearly values
                data = {
                    'Year': t,
                    'Manure': C[0],
                    'Straw': C[1],
                    'Roots': C[2],
                    'Old': C[3],
                    'Total_ICBM': np.sum(C, axis=0)
                }
                
                global df_R2
                df_R2 = pd.DataFrame(data)
                return df_R2
                            
                

            test()
            
    
            
            df_R2['Stable_Biochar'] = biochar_R1

            df_R2['Cumulative_Biochar'] = df_R2['Stable_Biochar'].cumsum()
            df_R2['Total_Carbon'] = df_R2['Total_ICBM'] + df_R2['Cumulative_Biochar']

            
            
            # Assuming df_R5 has 30 rows
            df_R2['Year'] = range(0, 50)

            # Setting 'Year' column as the index
            df_R2.set_index('Year', inplace=True)
            
            
            df_R2_save = pd.DataFrame(df_R2)

            df_R2_save = df_R2_save.iloc[:30]
            
            

                
            print(df_R2_save)

        ICBM_third_and_Final_good_model()
        
        
        ''' ROTATION 3 with all the inputs '''
        
        print("********************")
        print("********************")
        print("********************")
        print("********************")
        print("HERE IS EVERYTHING REGARDING ROTATION 3")
        print("HERE IS EVERYTHING REGARDING ROTATION 3")
        print("********************")
        print("********************")
            
        def ICBM_third_and_Final_good_model():
            
            R1_import = pd.DataFrame(new_df_cereal3)
            
            print("TITTAR PÅ R1 här innan assignment ")
            print(R1_import)

            
            def importing_added_rotation_carbon():
                ''''''
                
                global straw_R1
                straw_R1 = R1_import.iloc[0, 1]  # iloc[0, 1] refers to the first row and second column.

                
                global roots_R1
                roots_R1 = R1_import.iloc[0, 2]
                
                global manure_R1
                manure_R1 = R1_import.iloc[0, 3]
                
                global stable_CC_R1
                stable_CC_R1 = R1_import.iloc[0, 5]
                
                global biochar_R1
                biochar_R1 = R1_import.iloc[0, 4] 
                
                print(straw_R1)
                print(roots_R1)
                print(manure_R1)
                
                
                
            importing_added_rotation_carbon()
            
            
            
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib.pyplot as plt

            class model():
                def __init__(self, para, ini):
                    self.ky = para['ky']    
                    self.ko = para['ko']    
                    self.hm = para['hm']    
                    self.hs = para['hs']
                    self.hr = para['hr']
                    
                    self.Y_manure = ini['Y_manure']
                    self.Y_straw = ini['Y_straw']
                    self.Y_roots = ini['Y_roots']
                    self.O = ini['O']

                def compute(self, t, I_manure=0.0, I_straw=0.0, I_roots=0.0, Re=1):
                    
                    t = np.array(t, ndmin=1)
                    I_manure = I_manure*np.ones(np.shape(t))
                    I_straw = I_straw*np.ones(np.shape(t))
                    I_roots = I_roots*np.ones(np.shape(t))
                    Re = Re*np.ones(np.shape(t))
                    
                    nsteps = len(t)
                    C = np.zeros((4, nsteps))
                    C[0,0] = self.Y_manure
                    C[1,0] = self.Y_straw
                    C[2,0] = self.Y_roots
                    C[3,0] = self.O

                    for m in range(1,nsteps):
                        k = [self.ky * Re[m], self.ko * Re[m]]
                        h = [self.hm, self.hs, self.hr]
                        x0 = [self.Y_manure, self.Y_straw, self.Y_roots, self.O]
                        I = [I_manure[m], I_straw[m], I_roots[m]]
                        tspan = [t[m-1], t[m]]
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        for i in range(4):
                            C[i,m] = x[1,i]
                        self.Y_manure = x[1,0]
                        self.Y_straw = x[1,1]
                        self.Y_roots = x[1,2]
                        self.O = x[1,3]
                        
                    transfer_rates = []  # List to store the transfer rates from Y_straw to O at each time step
                    
                    for m in range(1,nsteps):
                        ...
                        # Check if the current year (or time step) is a 6th year
                        current_year = t[m]
                        if current_year % 6 == 0:
                            I_roots[m] += stable_CC_R1  # Add the Stable_CC to the Y_roots input
                        
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        transfer_rate = h[1] * k[0] * x[1, 1]  # h[1]*k[0]*Y_straw
                        transfer_rates.append(transfer_rate)
                        ...

                                                            
                    
                    print("Size of Y_straw at each time step:", C[1])
                    print("Transfer rate from Y_straw to O at each time step:", transfer_rates)
                        
                        
                        
                    return C

            def time_derivative(x, t, I, k, h):
                dy_manure_dt = I[0] - k[0]*x[0]
                dy_straw_dt = I[1] - k[0]*x[1]
                dy_roots_dt = I[2] - k[0]*x[2]
                do_dt = h[0]*k[0]*x[0] + h[1]*k[0]*x[1] + h[2]*k[0]*x[2] - k[1]*x[3]
                return [dy_manure_dt, dy_straw_dt, dy_roots_dt, do_dt]
            
            def test():
                # test ICBM for winter wheat field
                para = {
                        'ky': 0.758,     # yr-1
                        'ko': 6.05e-3,   # yr-1
                        'hm': 0.32,      # -
                        'hs': 0.2,       # -
                        'hr': 0.39       # -
                        }
                            
                ini = {
                    'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                    'Y_straw': straw_R1,
                    'Y_roots': roots_R1,
                    'O': SOC_init_cereal3 
                    }


                
                added_C = {
                        'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                        'Y_straw': straw_R1,
                        'Y_roots': roots_R1
                        }
                            
                # create model instance
                run1 = model(para, ini)
                
                n = 30  # yr
                t = np.linspace(0, n)
                
                # run model and return results
                C = run1.compute(t, I_straw=added_C['Y_straw'],  I_roots=added_C['Y_roots'], Re=0.8)
                
                
                
                
                total_initial = np.sum(list(ini.values()))
                total_final = np.sum(C[:, -1])
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100
                
                print(f"Total accumulated added straw over 30 years: {increase_amount:.2f}")
                print(f"Amount transferred from the young straw pool to the old pool: {increase_percent:.2f}")

                # plot figure
                plt.figure()

                #plt.plot(t, C[0], label='Manure')
                plt.plot(t, C[1], label='Straw')
                plt.plot(t, C[2], label='Roots')

                plt.plot(t, C[3], label='Old')
                plt.plot(t, np.sum(C, axis=0), label='Total')
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Standrad rotation R3')
                plt.savefig('ICBM_test.png', dpi=300)
                
                #plt.show()
                
                # Calculate increase in the total pool after 30 years
                total_initial = np.sum(list(ini.values()))

                total_final = np.sum(C, axis=0)[-1]
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100

                print(f"Increase in total pool after 30 years: {increase_amount:.2f}")
                print(f"Percentage increase: {increase_percent:.2f}%")
                
                # Extract final values after 30 years
                final_manure = C[0, -1]
                final_straw = C[1, -1]
                final_roots = C[2, -1]
                final_old_pool = C[3, -1]
                final_total = np.sum(C[:, -1])

                print(f"Manure pool after 30 years: {final_manure:.2f}")
                print(f"Straw pool after 30 years: {final_straw:.2f}")
                print(f"Roots pool after 30 years: {final_roots:.2f}")
                print(f"Old pool after 30 years: {final_old_pool:.2f}")
                print(f"Total after 30 years: {final_total:.2f}")
                
                
                # Creating a DataFrame to store the yearly values
                data = {
                    'Year': t,
                    'Manure': C[0],
                    'Straw': C[1],
                    'Roots': C[2],
                    'Old': C[3],
                    'Total_ICBM': np.sum(C, axis=0)
                }
                
                global df_R3
                df_R3 = pd.DataFrame(data)
                return df_R3
                            
                

            test()
            
    
            
            df_R3['Stable_Biochar'] = biochar_R1

            df_R3['Cumulative_Biochar'] = df_R3['Stable_Biochar'].cumsum()
            df_R3['Total_Carbon'] = df_R3['Total_ICBM'] + df_R3['Cumulative_Biochar']
            
            
            # Assuming df_R5 has 30 rows
            df_R3['Year'] = range(0, 50)

            # Setting 'Year' column as the index
            df_R3.set_index('Year', inplace=True)
            
            df_R3_save = pd.DataFrame(df_R3)

            df_R3_save = df_R3_save.iloc[:30]
            
            

                
            print(df_R3_save)

        ICBM_third_and_Final_good_model()
            
        
        
        #ÄNDRA INITIAL CARBON FÖR 4 och 5
        
        
        ''' ROTATION 4 with all the inputs '''
        
        print("********************")
        print("********************")
        print("********************")
        print("********************")
        print("HERE IS EVERYTHING REGARDING ROTATION 4")
        print("HERE IS EVERYTHING REGARDING ROTATION 4")
        print("********************")
        print("********************")
            
        def ICBM_third_and_Final_good_model():
            
            R1_import = pd.DataFrame(new_df_cereal4)
            
            print("TITTAR PÅ R1 här innan assignment ")
            print(R1_import)

            
            def importing_added_rotation_carbon():
                ''''''
                
                global straw_R1
                straw_R1 = R1_import.iloc[0, 1]  # iloc[0, 1] refers to the first row and second column.

                
                global roots_R1
                roots_R1 = R1_import.iloc[0, 2]
                
                global manure_R1
                manure_R1 = R1_import.iloc[0, 3]
                
                global stable_CC_R1
                stable_CC_R1 = R1_import.iloc[0, 5]
                
                global biochar_R1
                biochar_R1 = R1_import.iloc[0, 4] 
                
                print(straw_R1)
                print(roots_R1)
                print(manure_R1)
                
                
                
            importing_added_rotation_carbon()
            
            
            
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib.pyplot as plt

            class model():
                def __init__(self, para, ini):
                    self.ky = para['ky']    
                    self.ko = para['ko']    
                    self.hm = para['hm']    
                    self.hs = para['hs']
                    self.hr = para['hr']
                    
                    self.Y_manure = ini['Y_manure']
                    self.Y_straw = ini['Y_straw']
                    self.Y_roots = ini['Y_roots']
                    self.O = ini['O']

                def compute(self, t, I_manure=0.0, I_straw=0.0, I_roots=0.0, Re=1):
                    
                    t = np.array(t, ndmin=1)
                    I_manure = I_manure*np.ones(np.shape(t))
                    I_straw = I_straw*np.ones(np.shape(t))
                    I_roots = I_roots*np.ones(np.shape(t))
                    Re = Re*np.ones(np.shape(t))
                    
                    nsteps = len(t)
                    C = np.zeros((4, nsteps))
                    C[0,0] = self.Y_manure
                    C[1,0] = self.Y_straw
                    C[2,0] = self.Y_roots
                    C[3,0] = self.O

                    for m in range(1,nsteps):
                        k = [self.ky * Re[m], self.ko * Re[m]]
                        h = [self.hm, self.hs, self.hr]
                        x0 = [self.Y_manure, self.Y_straw, self.Y_roots, self.O]
                        I = [I_manure[m], I_straw[m], I_roots[m]]
                        tspan = [t[m-1], t[m]]
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        for i in range(4):
                            C[i,m] = x[1,i]
                        self.Y_manure = x[1,0]
                        self.Y_straw = x[1,1]
                        self.Y_roots = x[1,2]
                        self.O = x[1,3]
                        
                    transfer_rates = []  # List to store the transfer rates from Y_straw to O at each time step
                    
                    for m in range(1,nsteps):
                        ...
                        # Check if the current year (or time step) is a 6th year
                        current_year = t[m]
                        if current_year % 6 == 0:
                            I_roots[m] += stable_CC_R1  # Add the Stable_CC to the Y_roots input
                        
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        transfer_rate = h[1] * k[0] * x[1, 1]  # h[1]*k[0]*Y_straw
                        transfer_rates.append(transfer_rate)
                        ...

                                                            
                    
                    print("Size of Y_straw at each time step:", C[1])
                    print("Transfer rate from Y_straw to O at each time step:", transfer_rates)
                        
                        
                        
                    return C

            def time_derivative(x, t, I, k, h):
                dy_manure_dt = I[0] - k[0]*x[0]
                dy_straw_dt = I[1] - k[0]*x[1]
                dy_roots_dt = I[2] - k[0]*x[2]
                do_dt = h[0]*k[0]*x[0] + h[1]*k[0]*x[1] + h[2]*k[0]*x[2] - k[1]*x[3]
                return [dy_manure_dt, dy_straw_dt, dy_roots_dt, do_dt]
            
            def test():
                # test ICBM for winter wheat field
                para = {
                        'ky': 0.758,     # yr-1
                        'ko': 6.05e-3,   # yr-1
                        'hm': 0.32,      # -
                        'hs': 0.2,       # -
                        'hr': 0.39       # -
                        }
                            
                ini = {
                    'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                    'Y_straw': straw_R1,
                    'Y_roots': roots_R1,
                    'O': SOC_init_dairy 
                    }


                
                added_C = {
                        'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                        'Y_straw': straw_R1,
                        'Y_roots': roots_R1
                        }
                            
                # create model instance
                run1 = model(para, ini)
                
                n = 30  # yr
                t = np.linspace(0, n)
                
                # run model and return results
                C = run1.compute(t, I_straw=added_C['Y_straw'],  I_roots=added_C['Y_roots'], Re=0.8)
                
                
                
                
                total_initial = np.sum(list(ini.values()))
                total_final = np.sum(C[:, -1])
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100
                
                print(f"Total accumulated added straw over 30 years: {increase_amount:.2f}")
                print(f"Amount transferred from the young straw pool to the old pool: {increase_percent:.2f}")

                # plot figure
                plt.figure()

                #plt.plot(t, C[0], label='Manure')
                plt.plot(t, C[1], label='Straw')
                plt.plot(t, C[2], label='Roots')

                plt.plot(t, C[3], label='Old')
                plt.plot(t, np.sum(C, axis=0), label='Total')
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Standrad rotation R4')
                plt.savefig('ICBM_test.png', dpi=300)
                
                #plt.show()
                
                # Calculate increase in the total pool after 30 years
                total_initial = np.sum(list(ini.values()))

                total_final = np.sum(C, axis=0)[-1]
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100

                print(f"Increase in total pool after 30 years: {increase_amount:.2f}")
                print(f"Percentage increase: {increase_percent:.2f}%")
                
                # Extract final values after 30 years
                final_manure = C[0, -1]
                final_straw = C[1, -1]
                final_roots = C[2, -1]
                final_old_pool = C[3, -1]
                final_total = np.sum(C[:, -1])

                print(f"Manure pool after 30 years: {final_manure:.2f}")
                print(f"Straw pool after 30 years: {final_straw:.2f}")
                print(f"Roots pool after 30 years: {final_roots:.2f}")
                print(f"Old pool after 30 years: {final_old_pool:.2f}")
                print(f"Total after 30 years: {final_total:.2f}")
                
                
                # Creating a DataFrame to store the yearly values
                data = {
                    'Year': t,
                    'Manure': C[0],
                    'Straw': C[1],
                    'Roots': C[2],
                    'Old': C[3],
                    'Total_ICBM': np.sum(C, axis=0)
                }
                
                global df_R4
                df_R4 = pd.DataFrame(data)
                return df_R4
                            
                

            test()
            
    
            
            df_R4['Stable_Biochar'] = biochar_R1

            df_R4['Cumulative_Biochar'] = df_R4['Stable_Biochar'].cumsum()
            df_R4['Total_Carbon'] = df_R4['Total_ICBM'] + df_R4['Cumulative_Biochar']
            
            
            
            df_R4_save = pd.DataFrame(df_R4)

            df_R4_save = df_R4_save.iloc[:30]
            
            

                
            print(df_R4_save)

        ICBM_third_and_Final_good_model()  


        ''' ROTATION 5 with all the inputs '''
        
        print("********************")
        print("********************")
        print("********************")
        print("********************")
        print("HERE IS EVERYTHING REGARDING ROTATION 5")
        print("HERE IS EVERYTHING REGARDING ROTATION 5")
        print("********************")
        print("********************")
            
        def ICBM_third_and_Final_good_model():
            
            R1_import = pd.DataFrame(new_df_cereal5)
            
            print("TITTAR PÅ R1 här innan assignment ")
            print(R1_import)

            
            def importing_added_rotation_carbon():
                ''''''
                
                global straw_R1
                straw_R1 = R1_import.iloc[0, 1]  # iloc[0, 1] refers to the first row and second column.

                
                global roots_R1
                roots_R1 = R1_import.iloc[0, 2]
                
                global manure_R1
                manure_R1 = R1_import.iloc[0, 3]
                
                global stable_CC_R1
                stable_CC_R1 = R1_import.iloc[0, 5]
                
                global biochar_R1
                biochar_R1 = R1_import.iloc[0, 4] 
                
                print(straw_R1)
                print(roots_R1)
                print(manure_R1)
                
                
                
            importing_added_rotation_carbon()
            
            
            
            import numpy as np
            from scipy.integrate import odeint
            import matplotlib.pyplot as plt

            class model():
                def __init__(self, para, ini):
                    self.ky = para['ky']    
                    self.ko = para['ko']    
                    self.hm = para['hm']    
                    self.hs = para['hs']
                    self.hr = para['hr']
                    
                    self.Y_manure = ini['Y_manure']
                    self.Y_straw = ini['Y_straw']
                    self.Y_roots = ini['Y_roots']
                    self.O = ini['O']

                def compute(self, t, I_manure=0.0, I_straw=0.0, I_roots=0.0, Re=1):
                    
                    t = np.array(t, ndmin=1)
                    I_manure = I_manure*np.ones(np.shape(t))
                    I_straw = I_straw*np.ones(np.shape(t))
                    I_roots = I_roots*np.ones(np.shape(t))
                    Re = Re*np.ones(np.shape(t))
                    
                    nsteps = len(t)
                    C = np.zeros((4, nsteps))
                    C[0,0] = self.Y_manure
                    C[1,0] = self.Y_straw
                    C[2,0] = self.Y_roots
                    C[3,0] = self.O

                    for m in range(1,nsteps):
                        k = [self.ky * Re[m], self.ko * Re[m]]
                        h = [self.hm, self.hs, self.hr]
                        x0 = [self.Y_manure, self.Y_straw, self.Y_roots, self.O]
                        I = [I_manure[m], I_straw[m], I_roots[m]]
                        tspan = [t[m-1], t[m]]
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        for i in range(4):
                            C[i,m] = x[1,i]
                        self.Y_manure = x[1,0]
                        self.Y_straw = x[1,1]
                        self.Y_roots = x[1,2]
                        self.O = x[1,3]
                        
                    transfer_rates = []  # List to store the transfer rates from Y_straw to O at each time step
                    
                    for m in range(1,nsteps):
                        ...
                        # Check if the current year (or time step) is a 6th year
                        current_year = t[m]
                        if current_year % 6 == 0:
                            I_roots[m] += stable_CC_R1  # Add the Stable_CC to the Y_roots input
                        
                        x = odeint(time_derivative, x0, tspan, args=(I, k, h))
                        transfer_rate = h[1] * k[0] * x[1, 1]  # h[1]*k[0]*Y_straw
                        transfer_rates.append(transfer_rate)
                        ...

                                                            
                    
                    print("Size of Y_straw at each time step:", C[1])
                    print("Transfer rate from Y_straw to O at each time step:", transfer_rates)
                        
                        
                        
                    return C

            def time_derivative(x, t, I, k, h):
                dy_manure_dt = I[0] - k[0]*x[0]
                dy_straw_dt = I[1] - k[0]*x[1]
                dy_roots_dt = I[2] - k[0]*x[2]
                do_dt = h[0]*k[0]*x[0] + h[1]*k[0]*x[1] + h[2]*k[0]*x[2] - k[1]*x[3]
                return [dy_manure_dt, dy_straw_dt, dy_roots_dt, do_dt]
            
            def test():
                # test ICBM for winter wheat field
                para = {
                        'ky': 0.758,     # yr-1
                        'ko': 6.05e-3,   # yr-1
                        'hm': 0.32,      # -
                        'hs': 0.2,       # -
                        'hr': 0.39       # -
                        }
                            
                ini = {
                    'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                    'Y_straw': straw_R1,
                    'Y_roots': roots_R1,
                    'O': SOC_init_dairy2 
                    }


                
                added_C = {
                        'Y_manure': manure_R1,          # Based on kg from my constructed model - and calculating backwards using the icbm numbers
                        'Y_straw': straw_R1,
                        'Y_roots': roots_R1
                        }
                            
                # create model instance
                run1 = model(para, ini)
                
                n = 30  # yr
                t = np.linspace(0, n)
                
                # run model and return results
                C = run1.compute(t, I_straw=added_C['Y_straw'],  I_roots=added_C['Y_roots'], Re=0.8)
                
                
                
                
                total_initial = np.sum(list(ini.values()))
                total_final = np.sum(C[:, -1])
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100
                
                print(f"Total accumulated added straw over 30 years: {increase_amount:.2f}")
                print(f"Amount transferred from the young straw pool to the old pool: {increase_percent:.2f}")

                # plot figure
                plt.figure()

                #plt.plot(t, C[0], label='Manure')
                plt.plot(t, C[1], label='Straw')
                plt.plot(t, C[2], label='Roots')

                plt.plot(t, C[3], label='Old')
                plt.plot(t, np.sum(C, axis=0), label='Total')
                plt.legend()
                plt.xlabel('Time (yr)')
                plt.ylabel('Carbon pools (kg C)')
                plt.title('Standrad rotation R5')
                plt.savefig('ICBM_test.png', dpi=300)
                
                #plt.show()
                
                # Calculate increase in the total pool after 30 years
                total_initial = np.sum(list(ini.values()))

                total_final = np.sum(C, axis=0)[-1]
                increase_amount = total_final - total_initial
                increase_percent = (increase_amount / total_initial) * 100

                print(f"Increase in total pool after 30 years: {increase_amount:.2f}")
                print(f"Percentage increase: {increase_percent:.2f}%")
                
                # Extract final values after 30 years
                final_manure = C[0, -1]
                final_straw = C[1, -1]
                final_roots = C[2, -1]
                final_old_pool = C[3, -1]
                final_total = np.sum(C[:, -1])

                print(f"Manure pool after 30 years: {final_manure:.2f}")
                print(f"Straw pool after 30 years: {final_straw:.2f}")
                print(f"Roots pool after 30 years: {final_roots:.2f}")
                print(f"Old pool after 30 years: {final_old_pool:.2f}")
                print(f"Total after 30 years: {final_total:.2f}")
                
                
                # Creating a DataFrame to store the yearly values
                data = {
                    'Year': t,
                    'Manure': C[0],
                    'Straw': C[1],
                    'Roots': C[2],
                    'Old': C[3],
                    'Total_ICBM': np.sum(C, axis=0)
                }
                
                global df_R5
                df_R5 = pd.DataFrame(data)
                return df_R5
                            
                

            test()
            
    
            
            df_R5['Stable_Biochar'] = biochar_R1

            df_R5['Cumulative_Biochar'] = df_R5['Stable_Biochar'].cumsum()
            df_R5['Total_Carbon'] = df_R5['Total_ICBM'] + df_R5['Cumulative_Biochar']
            
            
            
            
            # Assuming df_R5 has 30 rows
            df_R5['Year'] = range(0, 50)

            # Setting 'Year' column as the index
            df_R5.set_index('Year', inplace=True)
            
            df_R5_save = pd.DataFrame(df_R5)

            df_R5_save = df_R5_save.iloc[:30]
            
            
            

                
            print(df_R5_save)

        ICBM_third_and_Final_good_model()  

        
        # Assuming you have df_R1 to df_R5 already defined
        dfs = [df_R1, df_R2, df_R3, df_R4, df_R5]
        
       

        # Extract 'Total_Carbon' from each DataFrame and concatenate them
        
        global new_dataframe_with_all_SOC_balance
        
        new_dataframe_with_all_SOC_balance = pd.concat([df['Total_Carbon'] for df in dfs], axis=1)

        # Optionally, you can rename the columns for clarity
        new_dataframe_with_all_SOC_balance.columns = ['Total_Carbon_R1', 'Total_Carbon_R2', 'Total_Carbon_R3', 'Total_Carbon_R4', 'Total_Carbon_R5']

        print(new_dataframe_with_all_SOC_balance)  
        
        new_dataframe_with_all_SOC_balance = new_dataframe_with_all_SOC_balance.iloc[:30]


        # Print the new data frame
        print("THIS IS THE ONE FOR ALL THE DATA")
        print(new_dataframe_with_all_SOC_balance)



    
        


the_model_mechanics_and_functions_of_model()   
