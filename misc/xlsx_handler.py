"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



xlsx handler

Functions to handle saving data to .xlsx files

Credits:
    Saving array to .xlsx file
    LINK: https://stackoverflow.com/a/69258865

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
import os
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows

"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def save_dataframe_to_xlsx(array, index_name, column_names, result_file, sheet_name):
    # :: Function saves given array to .xlsx file
    # :: Credits: https://stackoverflow.com/a/69258865
    
    # Create dataframe from array
    df = pd.DataFrame(array,
                      index=[index_name],
                      columns=column_names)
    
    # Check if .xlsx file already exists
    if not os.path.exists(result_file):
        df.to_excel(result_file, sheet_name = sheet_name)
    else:
        workbook = openpyxl.load_workbook(result_file)  # load workbook if already exists
        sheet = workbook[sheet_name]  # declare the active sheet 
        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index=[index_name]):
            sheet.append(row)
        workbook.save(result_file)  # save workbook
        workbook.close()  # close workbook

def save_errors_to_xlsx(mean_array, variance_array, object_name, result_file = "saved_results"):
    # :: Function saves array of mean errors and variances to .xlsx file with desired name
    
    # Transpose arrays
    mean_array_tp = np.transpose(mean_array)
    var_array_tp  = np.transpose(variance_array)
    
    # Column names (error names)
    column_names = ['RE (°)','RE (mm)','RMSE (°)','RMSE (mm)','MAE (°)','MAE (mm)','Recall','R2','CD']
    
    # Save to .xlsx
    save_dataframe_to_xlsx(mean_array_tp, object_name, column_names, result_file + '_mean.xlsx', 'mean_results')
    save_dataframe_to_xlsx(var_array_tp, object_name, column_names, result_file + '_var.xlsx', 'variance_results')


def save_timing_to_xlsx(time_list,exp_name,result_file = "time_results.xlsx"):
    # :: Function saves timing data into .xlsx file
    # :: Credits: https://stackoverflow.com/a/69258865
    
    # Create array from list
    time_array = np.array(time_list)
    
    # Create dataframe from array
    df = pd.DataFrame(time_array[:,1],
                      index=[time_array[:,0]],
                      columns=[exp_name])
    
    # Check if .xlsx file already exists
    if not os.path.exists(result_file):
        df.to_excel(result_file, sheet_name='timing_sheet')
    else:
        workbook = openpyxl.load_workbook(result_file)  # load workbook if already exists
        sheet = workbook['timing_sheet']  # declare the active sheet 
        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index=[time_array[:,0]]):
            sheet.append(row)
        workbook.save(result_file)  # save workbook
        workbook.close()  # close workbook

def read_data(DIR,sht_name,nmb_rows,nmb_rows_skip,columns):
    # :: Function reads .xlsx file from given directory
    
    # Read dataframe with given columns/rows
    dataframe = pd.read_excel(DIR,sheet_name=sht_name,usecols=columns,nrows=nmb_rows,skiprows=nmb_rows_skip)
    
    return dataframe.values

def save_to_latex(DIR,sht_name,nmb_rows,nmb_rows_skip,columns, file_name):
    # :: Function reads .xlsx file and saves dataframe to .tex file
    
    # Read dataframe
    dataframe = pd.read_excel(DIR,sheet_name=sht_name,usecols=columns,nrows=nmb_rows,skiprows=nmb_rows_skip)
    print(dataframe)
    
    # Save to latex
    dataframe.to_latex(file_name,index=False,float_format="%.2f")
    # dataframe.to_latex('test.tex',index=False)
    return


# DIR = "C:/Users/menth/Documents/Universiteit/2223/Master Thesis/Results2.0_global_registration_BEST.xlsx"
# save_to_latex(DIR, "VARIANCE", 1, 72, "E:L,O")