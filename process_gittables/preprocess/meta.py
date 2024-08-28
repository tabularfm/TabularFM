import pandas as pd
import pickle
import yaml
import json

class Metadata():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.metadata = {}

    def handle_categorical(self, series: pd.Series, pii=False) -> dict:
        # NOTE: pii is removed in sdmetrics latest ver (0.9.0)
        # return dict(sdtype='categorical', pii=pii)
        return dict(sdtype='categorical')

    def handle_numerical(self, series: pd.Series, dtype:str) -> dict:
        if dtype == 'float':
            return dict(sdtype='numerical', subtype='float')

        if dtype == 'int':
            return dict(sdtype='numerical', subtype='integer')

    def handle_boolean(self, series: pd.Series) -> dict:
        return dict(sdtype='boolean')

    def handle_datetime(self, series: pd.Series) -> dict:
        #TODO: editable datetime format
        return dict(sdtype='datetime', format='%d-%m-%Y')
        # return dict(sdtype='datetime', format=None)
        # return dict(sdtype='categorical')

    def handle_id(self, series: pd.Series, subtype: str) -> dict:
        return dict(sdtype='id', subtype = 'string' if subtype == 'str' else 'integer')

    def to_pickle(self, filepath):
        """
        export to .pkl
        """
        pickle.dump(self.metadata, open(filepath, 'wb'))

    def to_yaml(self, filepath):
        yaml.dump(self.metadata, open(filepath, 'w'))
        
    def to_json(self, filepath):
        json.dump(self.metadata, open(filepath, 'w'))

class SingleTabMetadata(Metadata):
    """
    Automatically get metadata from tabular data
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super(SingleTabMetadata, self).__init__(df)

    def get_metadata(self, verbose:int =1 ) -> dict:
        self.metadata['columns'] = {}
        columns = self.df.columns
        for i, col in enumerate(columns):
            if verbose: print('col ', col)
            series = self.df[col]
            self.metadata['columns'][col] = self.get_meta_series(series)
            
            # if i == 0:
            #     self.metadata['primary_key'] = col
            #     self.metadata['columns'][col]['sdtype'] = "id"

        return self.metadata

    def get_meta_series(self, series: pd.Series) -> pd.DataFrame.dtypes:
        # numeric: int, float, id
       
        #   float
        if pd.api.types.is_float_dtype(series):
            return self.handle_numerical(series, dtype='float')

        #   int
        if pd.api.types.is_integer_dtype(series):
            #   id
            if series.is_unique:
                return self.handle_id(series, subtype='int')

            # ordinal int
            return self.handle_numerical(series, dtype='int')

        # bool
        if pd.api.types.is_bool_dtype(series):
            return self.handle_boolean(series)

        # object: str, categorical, datetime
        if pd.api.types.is_object_dtype(series):
            try:
                # whether it is datetime
                pd.to_datetime(series)
                return self.handle_datetime(series)
            except:
                # otherwise, it is categorical or str
                if series.is_unique: # str
                    return self.handle_categorical(series, pii=True)
                else: # ordinal categorical
                    return self.handle_categorical(series, pii=False)

    