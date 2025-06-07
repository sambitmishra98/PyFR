import h5py
from pyfr.inifile import Inifile
import pandas as pd
import re
from pyfr.plugins.base import (BaseCLIPlugin, cli_external)
class BenchmarkCLI(BaseCLIPlugin):
    name = 'benchmark'
    @classmethod
    def add_cli(cls, parser):
        sp=  parser.add_subparsers()
        bp = sp.add_parser('preprocess-configs', help='benchmark preprocess-configs --help')
        bp.set_defaults(process=cls.preprocess_configs_cli)
        bp.add_argument('--config', nargs='*', 
                        help='config file in inifile format', required=True)
        bp.add_argument('--options', nargs='*', 
                        help='csv file format of section and options to modify in the config file', required=True)
        bp = sp.add_parser('preprocess-scripts', help='benchmark preprocess-scripts --help')
        bp.set_defaults(process=cls.preprocess_scripts_cli)
        bp.add_argument('--script', nargs='*', 
                        help='script file in terminal bash format', required=True)
        bp.add_argument('--options', nargs='*', 
                        help='csv file format of section and options to modify in the config file', required=True)
        bp = sp.add_parser('postprocess', 
                           help='benchmark postprocess --help')
        bp.set_defaults(process=cls.postprocess_cli)
        bp.add_argument('--options', nargs='*', 
                        help='options to include in the output', required=True)
        bp.add_argument('--files', nargs='*', 
                        help='solution files', required=True)
        bp.add_argument('--output', nargs='*', 
                        help='output file, in csv format', required=False)
    @cli_external
    def preprocess_configs_cli(self, args):
        # Get the base configuration file
        config = Inifile.load(args.config[0])
        # Get the options from the commandline, those following the --options
        df = pd.read_csv(args.options[0], sep=',', skipinitialspace=True,
                         comment='#')
        # Only consider the columns that start with 'script:'
        df = df.filter(regex='config:')
        df['file-name'] = df.apply(lambda row: args.config[0].split('/')[-1].split('.')[0]
                                   + '_'.join(
            [f'{column.split(":")[1]}-{value}' for column, value in row.items()])+'.ini', axis=1)
        df.set_index('file-name', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        for file_name in df.index:
            for column, value in df.loc[file_name].items():
                section, option = column.split(":")[1].split('_')
                config.set(section, option, value)
            config_as_string = config.tostr().encode('utf-8')
            with open(file_name, 'wb') as f:
                f.write(config_as_string)
    @cli_external
    def preprocess_scripts_cli(self, args):
        # Get the base configuration file as text file
        script = open(args.script[0], 'r').read()
        # Get the options from the commandline, those following the --options
        df = pd.read_csv(args.options[0], sep=',', skipinitialspace=True,
                         comment='#')
        # Only consider the columns that start with 'script:'
        df = df.filter(regex='script:')
        df['file-name'] = df.apply(lambda row: args.script[0].split('/')[-1].split('.')[0]
                                   + '_'.join(
            [f'{column.split(":")[1]}-{value}' for column, value in row.items()])+'.sh', axis=1)
        df.set_index('file-name', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        for file_name in df.index:
            for column, value in df.loc[file_name].items():

                # Format of column is script-{option}
                # In the file, it is export option=value
                option = column.split(':')[1]
                script = re.sub(
                    rf'{option}=.+?(\s|$)', 
                     f'{option}={value}\n', 
                                script)

            script_as_string = script.encode('utf-8')   

            with open(file_name, 'wb') as f:
                f.write(script_as_string)
    @cli_external
    def postprocess_cli(self, args):
        # Get the options from the commandline, those following the --options
        df = self.pyfrs_to_csv(args.files)
        self.process_columns(df, include_if_present=['file-name', *args.options])
        print(df.to_string(index=False))
        # Save as a csv file    
        if args.output:
            df.to_csv(args.output[0], index=False)
    def inifile_as_dict(self, config_file):
        stats_dict = {}
        for section in config_file.sections():
            for option in config_file.items(section):
                stats_dict[f'{section}_{option}'] = config_file.get(section, option)
        return stats_dict
    def pyfrs_to_csv(self, files):
        df = pd.DataFrame()
        for file in files:
            with h5py.File(file, 'r') as f:
                config = f['config']
                stats  = f['stats']
                config = Inifile(config[()].decode('utf-8'))
                stats  = Inifile(stats [()].decode('utf-8'))
                # Add prefix of all config entries as 'config:'
                dict_config = self.inifile_as_dict(config)
                prefixed_dict_config = {}
                for key, value in dict_config.items():
                    prefixed_dict_config[f'config:{key}'] = value

                # Add prefix of all stats entries as 'stats:'
                dict_stats = self.inifile_as_dict(stats)
                prefixed_dict_stats = {}
                for key, value in dict_stats.items():
                    prefixed_dict_stats[f'stats:{key}'] = value

                # Save all stats entries in the dataframe, with column name as {section}_{option}
                # Create a dictionary from stats, then append to the dataframe
                stats_dict = {}
                stats_dict['file-name'] = file  
                # Combine
                df_dict = {**prefixed_dict_config, **prefixed_dict_stats}
                stats_dict.update(df_dict)
                df = df._append(stats_dict, ignore_index=True)
        return df
    def remove_columns_if_identical(self, df):
        for column in df.columns:
            if df[column].nunique() == 1:
                df.drop(column, axis=1, inplace=True)
        return df
    def process_columns(self, df, include_if_present=[]):
        columns_matched = [column for column in df.columns for expression in include_if_present if re.match(expression, column)]
        # Remove columns that are not matched
        for column in df.columns:
            if column not in columns_matched:
                df.drop(column, axis=1, inplace=True)
        return df