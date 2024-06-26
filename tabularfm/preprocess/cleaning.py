import pandas as pd
import numpy as np

class TabCleaning():
    def __init__(self, exclude=[], remove=[]):
        """AI is creating summary for __init__

        Args:
            exclude: exclude these columns in automated cleaning
            remove: remove these columns when cleaning 
        """
        self.exclude_cols = exclude
        self.remove_cols = remove
        self.process_cols = {} # dict with bool, True if col is kept, False if otherwise
    
    def is_timestamp(self, series):
        # # pandas < v2.2
        # if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(series):
        #     return True
        
        # pandas > v2.2
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series) or pd.api.types.is_timedelta64_ns_dtype(series):
        # if pd.api.types.is_datetime64_dtype(series):
        #     print(series)
        #     print(' --> ', pd.api.types.is_datetime64_dtype(series))
            return True
        
        return False
            
    def is_low_frequency_categorical_col(self, series, min_freq_threshold=0.08, pct_to_remove=0.8):
        """
        Detect low frequency of categories (e.g. columns with ids, addresses, etc.)
        Args:
            series: `Series` pandas, data input
            min_freq_threshold: `float`, default is 0.08. Unless maximum category frequency is equal or greater than the threhold, the column will be removed
            pct_to_remove: `float`, percentage compare to number of values to remove. If the number of categories is equal or greater than pct*number of data, it will be removed
        """
        if pd.core.dtypes.common.is_string_dtype(series) or pd.core.dtypes.common.is_integer_dtype(series):
            
            frequency = series.value_counts().to_dict()
            frequency = {k : v / len(series.values) for k,v in frequency.items()}
            
            # no category
            if len(list(frequency.values())) == 0:
                return True
            
            # only 1 category
            if len(list(frequency.values())) == 1:
                return True
            
            # too many categories, and very few samples per category
            if len(frequency) >= pct_to_remove * len(series):
                return True
            
            # the smallest category has number of samples under threshold
            if min(list(frequency.values())) < min_freq_threshold:
                return True
            
            
        return False
    
    def is_nan(self, series):
        # print('\t Check is na: ')
        if series.isna().any():
            # print('\t TRUE')
            return True
        
        # print('\t FALSE')
        return False
    
    def is_inf(self, series):
        try:
            if any(np.isinf(series)):
                return True
            
            return False
        
        except:
            return False
    
    
    def fill_na(self, series):
        if series.dtype == object or series.dtype == int:
            max_value_to_fill = series.value_counts().max()
            series = series.fillna(max_value_to_fill)
        else:
            mean = series[series.isna() == False].mean()
            series = series.fillna(mean)
        
        return series
    
    def is_id_column(self, series):
        if len(series.unique()) == len(series):
            return True

        return False
            
    def clean(self, df, remove_timestamp=True, remove_low_frequency=True, remove_id=True, verbose=1, return_info=False, **kwargs):
        """Clean data, consists of 
        - timestamp
        - low frequency (e.g. address, id, etc.)
        - remove nan:  
            if nan_pct > pct_to_remove then remove col
            if nan_pct > pct_to_remove_row then remove row
            else impute with mode (categorical), mean (numerical)
        - fill inf: the same as nan
        

        Args:
            df (_type_): _description_
            remove_timestamp (bool, optional): _description_. Defaults to True.
            remove_low_frequency (bool, optional): _description_. Defaults to True.
            remove_id (bool, optional): _description_. Defaults to True.
            verbose (int, optional): _description_. Defaults to 1.
            return_info (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        pct_to_remove_row = kwargs['pct_to_remove_row'] if 'pct_to_remove_row' in kwargs else 0.
        
        cleaning_info = {} # {col_name: {keep: True / False, desc: reason to remove}}
        
        columns = df.columns
        for col in columns:
            
            self.process_cols[col] = False # auto assign False
            if verbose: print('col ', col)

            if col in self.remove_cols:
                del df[col]
                cleaning_info[col] = {'keep': False, 'desc': 'IN_REMOVE_LIST'}
                if verbose: print('\t Removed')
                continue
            
            if col in self.exclude_cols:
                cleaning_info[col] = {'keep': True, 'desc': 'IN_EXCLUDE_LIST'}
                if verbose: print('\t Excluded ')
                continue
            
            series = df[col]
            
            # ID COLUMN
            if self.is_id_column(series):
                if verbose: print('\t del: id column')
                del df[col]
                cleaning_info[col] = {'keep': False, 'desc': 'ID'}
                continue
            
            # REMOVE
            if remove_timestamp:
                if self.is_timestamp(series):
                    if verbose: print('\t del: timestamp datatype')
                    del df[col]
                    cleaning_info[col] = {'keep': False, 'desc': 'TIMESTAMP'}
                    continue
            
            if remove_low_frequency:
                min_freq_threshold = kwargs['min_freq_threshold'] if 'min_freq_threshold' in kwargs else 0.08
                pct_to_remove = kwargs['pct_to_remove'] if 'pct_to_remove' in kwargs else 0.8
                if self.is_low_frequency_categorical_col(series, min_freq_threshold=min_freq_threshold, pct_to_remove=pct_to_remove):
                    if verbose: print('\t del: low frequency values or id column')
                    del df[col]
                    cleaning_info[col] = {'keep': False, 'desc': 'LOW_FREQUENCY'}
                    continue
                
            # UPDATE SERIES
            # FILL NA
            if self.is_nan(series):
                # calculate percentage of nan
                nan_pct = series.isna().sum() / len(df)
                
                # print('\t nan pct: ', nan_pct)
                
                if nan_pct >= pct_to_remove: # if nan very large, then remove the series
                    if verbose: print('\t del: large nan values')
                    del df[col]
                    cleaning_info[col] = {'keep': False, 'desc': 'LARGE_NAN'}
                    continue
                
                elif nan_pct >= pct_to_remove_row:
                # elif series.dtype == object: # remove nan (row)
                    df.dropna(subset=[series.name], inplace=True)
                    
                else: # if float, int, ...
                    series = self.fill_na(series)
                    # df[col] = series  
                
            # FILL INF
            if self.is_inf(series):
                # calculate percentage of nan
                inf_pct = np.isinf(series).sum() / len(df)
                
                # print('\t inf pct: ', inf_pct)
                
                if inf_pct >= pct_to_remove: # if nan very large, then remove the series
                    if verbose: print('\t del: large inf values')
                    del df[col]
                    cleaning_info[col] = {'keep': False, 'desc': 'LARGE_INF'}
                    continue
                
                elif inf_pct >= pct_to_remove_row:
                    series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    # Drop rows with NaN
                    series.dropna(inplace=True)
                # elif series.dtype == object: # remove inf (row)
                #     # Replace infinite updated data with nan
                #     series.replace([np.inf, -np.inf], np.nan, inplace=True)
                #     # Drop rows with NaN
                #     series.dropna(inplace=True)
                    
                else: # if float, int, ...
                    series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    series = self.fill_na(series)
                    
            df[col] = series
            
            # -----------
            self.process_cols[col] = True #

            cleaning_info[col] = {'keep': True, 'desc': 'NA'}
            
            if verbose: print('\t pass')
        
        if not return_info:   
            return df

        return df, cleaning_info