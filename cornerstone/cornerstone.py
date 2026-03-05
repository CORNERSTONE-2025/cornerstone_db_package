import os
import h5py
import time
import numpy as np
import sys
import pandas as pd
#import mysql.connector
import gzip
import subprocess
import requests
import json
import re
import tempfile

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Dict
from types import SimpleNamespace
from datetime import datetime
import gdown 
import signal
import sqlite3
import shutil
 
import clearml
from clearml import Task
import torch
import argparse
import importlib.util
import sys
from datetime import datetime
from .ml.utils.training import run_cv_training, run_test
from .ml.utils.plots import plot_boxplot, plot_training_curves, plot_metric_curves, plot_test_metrics
from .ml.utils.path_manager import PathManager
from .ml.utils.model_manager import ModelManager
from .ml.utils.metrics_manager import MetricsManager
from .ml.utils.config_manager import ConfigManager

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import delle utilities usate nella pipeline di splitting/filtering
from sharp_data import (
    SharpDataset,
    FeatureSelector,
    MissingnessFilter,
    MultiWindowMissingnessFilter,
    MultiWindowEmptyFilter,
    TimeWindowSelector,
    TimeWindowFilter,
    TimeWindowExcluder,
    CMASKFilter,
    Compose,
    SharpDatasetSplitter,
    save_split_to_h5
)
__all__ = ["Cornerstone"]

class Cornerstone:
    """
    Cornerstone
    ------------
    SQL Database Handler for Solar Physics Datasets.

    Provides functionality to:
    - Connect to a MySQL database
    - Extract SHARP parameter time series for solar flare events
    - Save datasets to HDF5 format
    - Run operations in parallel using multiprocessing

    Author: Simone Chierichini
    """

    def __init__(self, db_config, output_folder="./dataset_compilation/data", project_name="Cornerstone"):
        
        self.db_config = db_config # TO-DO METTERE DENTRO PARTE DI CONFIG (usare solo sqlite3) 
        
        self.output_folder = output_folder # TO-DO METTERE DENTRO CLEARML
        self.rr = clearml #TO-DO METTERE COME PRIVATA O PROTETTA   # rr stands for reproducible research 
        self.project_name = project_name # TO-DO METTERE DENTRO CLEARML
        self.tasks = {} # TO-DO METTERE DENTRO CLEARML
        
        #IDEALMENTE USARE QUESTA STRUTTURA 
        '''
        self.config = load_config(config)
        self.database = CornerstoneDatabase(self.config)
        self.dataset = CornerstoneDataset(self.config)
        self.model = CornerstoneModel(self.config)
        self.plot = CornerstonePlot(self.config)
        '''
        
        os.makedirs(self.output_folder, exist_ok=True)
 
    def db_update(self):
        """
        Checks the local database for updates and synchronizes it with the remote database.
        
        Workflow:
        1. Load local metadata and fetch remote metadata.
        2. If no local database exists, download the first full dump and create the local DB.
        3. If remote DB has updates newer than the last sync, apply incremental updates.
        4. Update local metadata after synchronization.
        """
        # Create a task for logging/monitoring
        task = _get_or_create_task(self, "DB Update")
        print("🔍 Checking for database updates ...")

        # Load metadata about the local DB and fetch remote metadata
        local_meta = _load_local_metadata(self) 
        remote_meta = _fetch_remote_metadata(self)

        # Parse timestamps for full dump and latest update
        latest_full = datetime.fromisoformat(remote_meta["latest_full"])
        latest_update = datetime.fromisoformat(remote_meta["latest_update"])
        
        # -----------------------------
        # Case 0: No local DB exists
        # -----------------------------
        if local_meta["last_sync"] == "":
            print(f"🆕 Downloading first dump available from {remote_meta['latest_full']}.")
            db_folder_url = self.db_config["remote_db_url"]  
            
            # Download the full database dump
            full_path = _download_file(self, db_folder_url )       #full_path = "/bigdisk/daniele/cornerstone_module/db/cornerstone_full_2025-11-01.sql.gz"
            
            # Create local database from downloaded dump
            _create_local_db_from_dump(self, full_path, db_path = local_meta["db_path"] )
             
            # Save metadata for the downloaded full dump
            _save_local_metadata(self, remote_meta["latest_full"])
            
            # Refresh in memory metadata 
            local_meta = _load_local_metadata(self)
            print(f"🎉 Local database refreshed to full version {remote_meta['latest_full']}.")
                
        # -----------------------------
        # Case 1: Remote DB has updates newer than local
        # -----------------------------
        if  latest_update > datetime.fromisoformat(local_meta["last_sync"]):
            print(datetime.fromisoformat(local_meta["last_sync"])) 
            
            # Identify incremental updates newer than last sync
            updates_to_apply = [
                fname for fname in remote_meta["available_updates"]
                if datetime.fromisoformat(fname.split("_")[1]) > datetime.fromisoformat(local_meta["last_sync"])
            ] 
            print(f"📂 Found {len(updates_to_apply)} incremental updates: {updates_to_apply}")            
            
            # Apply each incremental update
            for update in updates_to_apply :
                url = remote_meta["available_updates"][update]
                 
                path = _download_file(url)
                _apply_update_to_local_db(self, local_meta["db_path"]+"/cornerstone.db" , path)
 
            # Save metadata after updates 
            _save_local_metadata(self, remote_meta["latest_update"]) 
            
            print(f"🎉 Local database refreshed to full version {remote_meta['latest_full']}.")
            
        # -----------------------------
        # Case 2: Local DB already up-to-date
        # -----------------------------
        else :
            task.close()
            print("✅ Local database already up to date.")
            return
        task.close()
        print(f"🎉 Database successfully updated to {remote_meta['latest_update']}.")


    def splitting_labeling(self, shtyp = "DEF", class0 = "Cplus",delta_hours="24", db_type = "sqlite", save_csv= False ):
        ##############################################################################################################
        ######################################################
        ######################################################
        # cut0
        #            'X': 1e-4,
        #            'M': 1e-5,
        #            'C': 1e-6,
        #DP : end and start for dddd in pd.date_range now are fixed
        
        # Create a task for logging/monitoring
        task = _get_or_create_task(self, "Splitting labeling") 
        
        CUT_C = 1e-6
        CUT_M = 1e-5
        CUT_X = 1e-4

        if class0=="Xplus":
            cut0=CUT_X       
        elif class0=="Mplus":
            cut0=CUT_M  
        elif class0=="Cplus":
            cut0=CUT_C
            
        header_flag=True
        step_freq=delta_hours+"h"
        delt_before=delta_hours+":00:00"          #### hh:mm:ss #### abs_time= t_X-delt_before
        delt_after =delta_hours+":00:00"          #### hh:mm:ss #### abs_time= t_X+delt_after
        delt_after_after =delta_hours+":00:00"    #### hh:mm:ss #### abs_time= t_X+delt_after+delt_after_after


        if db_type =="sqlite":
            
            import sqlite3 
            from sqlalchemy import create_engine

            # Step 1 — Ensure database existsù
            db_path = self.db_config["local_db_path"] 
            try:
                conn = sqlite3.connect(db_path)
            except Exception as err:
                print(f"❌ SQLite error: {err}", flush=True)

            
            # Load harp data once for efficiency
            query_harps = f"SELECT DISTINCT HARPNUM, MIN(T_OBS) AS startt, MAX(T_OBS) AS endt FROM SHARPC_all_{shtyp} GROUP BY HARPNUM"
            df_harps = pd.read_sql_query(query_harps, conn)
            df_harps['startt'] = pd.to_datetime(df_harps['startt'],
                                                format='%Y.%m.%d_%H:%M:%S_TAI',
                                                errors='coerce'           # ← very useful: turns bad dates into NaT instead of crashing
                                                ) 
            df_harps['endt'] = pd.to_datetime(df_harps['endt'],
                                                format='%Y.%m.%d_%H:%M:%S_TAI',
                                                errors='coerce'           # ← very useful: turns bad dates into NaT instead of crashing
                                                )

            # Load flare data once for efficiency (assuming fflux is numeric; adjust if it's string)
            query_flares = f"SELECT * FROM NOAA_SHARPC_{shtyp}_AR WHERE fflux >= {cut0}"#
            df_flares = pd.read_sql_query(query_flares, conn)
            
            query_flares_u = f"SELECT * FROM NOAA_SHARPC_{shtyp}_AR WHERE fflux >= {CUT_C}"#
            df_flares_u = pd.read_sql_query(query_flares_u, conn)
             
           
            df_flares['tmax'] = pd.to_datetime(df_flares['tmax'] ) 
            
            df_flares_u['tmax'] = pd.to_datetime(df_flares_u['tmax'] ) 
            # Close the connection as data is now in memory
            conn.close()

            # Assuming header_flag is initialized to True before the loop
            header_flag = True  # If not already defined

            for dddd in pd.date_range(start=pd.to_datetime("1996-01-01T00:00:00"), end=pd.to_datetime("2025-04-30T00:00:00"), freq=step_freq):
                result = pd.DataFrame()
                t_X_dt = dddd
                t_X = t_X_dt.strftime("%Y-%m-%dT%H:%M:%S")
                print('#############' + t_X + '##############')
                t_before_dt = t_X_dt - pd.Timedelta(delt_before)
                t_after_dt = t_X_dt + pd.Timedelta(delt_after)
                t_after_after_dt = t_after_dt + pd.Timedelta(delt_after_after)
                t_before = t_before_dt.strftime("%Y-%m-%dT%H:%M:%S")
                t_after = t_after_dt.strftime("%Y-%m-%dT%H:%M:%S")
                t_after_after = t_after_after_dt.strftime("%Y-%m-%dT%H:%M:%S") 

                # Filter active regions in pandas
                active_regions = df_harps[(df_harps['startt'] <= t_X_dt) & (df_harps['endt'] >= t_X_dt)]
 
                if not active_regions.empty:
                    for idx, row in active_regions.iterrows():
                        reg = row['HARPNUM']

                        # Filter flares for this region once
                        reg_flares = df_flares[df_flares['HARPNUM'] == reg]
                        
                        reg_flares_u = df_flares_u[df_flares_u['HARPNUM'] == reg]
                        

                        # Count before (inclusive: >= t_before_dt <= t_X_dt)
                        flares_b = reg_flares[(reg_flares['tmax'] >= t_before_dt) & (reg_flares['tmax'] <= t_X_dt) & reg_flares['fflux'] >= cut0]
                        nb = len(flares_b)
                        C_b = 1 if nb > 0 else 0

                        # Count after (inclusive: >= t_X_dt <= t_after_dt)
                        flares_a = reg_flares[(reg_flares['tmax'] >= t_X_dt) & (reg_flares['tmax'] <= t_after_dt)]
                        na = len(flares_a)
                        C_a = 1 if na > 0 else 0

                        flares_a_u = reg_flares_u[(reg_flares_u['tmax'] >= t_X_dt) & (reg_flares_u['tmax'] <= t_after_dt)]

                        
                        # Count after_after (inclusive: >= t_after_dt <= t_after_after_dt)
                        flares_aa = reg_flares[(reg_flares['tmax'] >= t_after_dt) & (reg_flares['tmax'] <= t_after_after_dt)]
                        naa = len(flares_aa)
                        C_aa = 1 if naa > 0 else 0

                        # REGION LABEL
                        label = "---"
                        label_u = "---"
                        if not C_b and not C_a and not C_aa:
                            label = "NO1"
                            label_u = "NO1"
                        if not C_b and not C_a and C_aa:
                            label = "NO2" 
                            label_u = "NO2"
                        if C_b and not C_a and C_aa:
                            label = "NO3"
                            label_u = "NO3" 
                        if C_b and not C_a and not C_aa:
                            label = "NO4" 
                            label_u = "NO4"
                        if C_a:
                            label = "YES" 
                            max_flux = flares_a_u['fflux'].max()
                            label_u = _flare_class_from_flux(max_flux)
                            
                        # Calculate hours
                        hours_staobs = (row['startt'] - t_X_dt).total_seconds() / 3600
                        hours_endobs = (row['endt'] - t_X_dt).total_seconds() / 3600

                        # Convert datetimes to strings for output consistency
                        start_obs = row['startt'].strftime("%Y-%m-%dT%H:%M:%S")
                        end_obs = row['endt'].strftime("%Y-%m-%dT%H:%M:%S")

                        x = [[t_X,  reg, label, label_u, C_b, C_a, C_aa, nb, na, naa, start_obs, hours_staobs, end_obs, hours_endobs]]
                        result = result._append(pd.DataFrame(x))
 

                    print("***************** Mind you have to drop the table every run *************")

                    result.columns = ['t_X', 'HARPn', 'label', 'label_u', 'C_b', 'C_a', 'C_aa', 'n_b', 'n_a', 'n_aa', 'obs_start', 'obs_startH', 'obs_end', 'obs_endH']
  
                    try:
                        conn = sqlite3.connect(db_path)
                    except Exception as err:
                        print(f"❌ SQLite error: {err}", flush=True)
                    result.to_sql(name='DF_' + shtyp + '_' + class0 + '_delta' + delta_hours + '_three_classes', con=conn, if_exists="append", index=False) 
                    
                    conn.close()
                    
                    if save_csv:
                        if header_flag:
                            result.to_csv('DF_' + shtyp + '_' + class0 + '_delta' + delta_hours + '_three_classes.csv', index=False, mode="a", header=True)
                            header_flag = False
                        else:
                            result.to_csv('DF_' + shtyp + '_' + class0 + '_delta' + delta_hours + '_three_classes.csv', index=False, mode="a", header=False)
                    
                    
     
        elif db_type == "Mysql":
            import mysql
            import mysql.connector
            from sqlalchemy import create_engine  

            # Step 1 — Ensure database exists
            try:
                mydb = mysql.connector.connect(**self.db_config)
                
            except mysql.connector.Error as err:
                print(f"❌ MySQL error: {err}", flush=True)
                  
            
            

            ######################################################
            ##> for dddd in pd.date_range(start=pd.to_datetime("2012-09-01T00:00:00"),end=pd.to_datetime("2025-04-30T00:00:00"),freq=step_freq): ### NRT
            for dddd in pd.date_range(start=pd.to_datetime("2010-05-01T00:00:00"),end=pd.to_datetime("2025-04-30T00:00:00"),freq=step_freq): ### DEF
                result=pd.DataFrame()
                t_X=dddd.strftime("%Y-%m-%dT%H:%M:%S")
                print('#############'+t_X+'##############')
                t_before     =(pd.to_datetime(t_X)-pd.Timedelta(delt_before)).strftime("%Y-%m-%dT%H:%M:%S")        # strings for the SQL
                t_after      =(pd.to_datetime(t_X)+pd.Timedelta(delt_after)).strftime("%Y-%m-%dT%H:%M:%S")
                t_after_after=(pd.to_datetime(t_X)+pd.Timedelta(delt_after)+pd.Timedelta(delt_after_after)).strftime("%Y-%m-%dT%H:%M:%S")
                
                
                query=   "select distinct HARPNUM," 
                query+=  " startt,time_to_sec(timediff(timestamp(startt),timestamp('"+t_X+"')))/3600 as hours_before," 
                query+=  " endt,  time_to_sec(timediff(timestamp(endt  ),timestamp('"+t_X+"')))/3600 as hours_after "
                query+=  " from (select HARPNUM,min(T_OBS) as startt,max(T_OBS) as endt from SHARPC_all_"+shtyp+" group by HARPNUM) as c "
                query+=  " where timestamp('"+t_X+"') between timestamp(c.startt) and timestamp(c.endt); "
            #    print(query)
                
                mycursor=mydb.cursor()
                mycursor.execute(query)
                myresult=mycursor.fetchall()
                regions=pd.DataFrame(myresult)
            ##############################################################################################################
                if len(regions)>0:
                    regions.columns=['HARPNUM','start_obs','hours_staobs','end_obs','hours_endobs']
                    for reg in regions['HARPNUM']:
                    #
                    #    print('>>>>>>>>>>>>>>>',reg,'<<<<<<<<<<<<<<<<<')
                    #
                        query_b ="select count(*),left(part1,1) from NOAA_SHARPC_"+shtyp+"_AR "           # b: t_before and t_X
                        query_b+="where HARPNUM="+str(reg)+" "
                        query_b+="and timestamp(tmax) between timestamp('"+t_before+"') and timestamp('"+t_X+"') "
                        query_b+="and fflux>='"+str(cut0)+"' "
                        mycursor=mydb.cursor()
                        mycursor.execute(query_b)
                        n_before=mycursor.fetchall()
                        nb=n_before[0][0]
                        C_b=0
                        if nb>0:
                            C_b=1
                    #
                        query_a ="select count(*),left(part1,1) from NOAA_SHARPC_"+shtyp+"_AR "           # a: after t_X, before t_after
                        query_a+="where HARPNUM="+str(reg)+" "
                        query_a+="and timestamp(tmax) between timestamp('"+t_X+"') and timestamp('"+t_after+"') "
                        query_a+="and fflux>='"+str(cut0)+"' "
                        mycursor=mydb.cursor()
                        mycursor.execute(query_a)
                        n_after=mycursor.fetchall()
                        na=n_after[0][0]
                        C_a=0
                        if na>0:
                            C_a=1
                    #
                        query_aa ="select count(*),left(part1,1) from NOAA_SHARPC_"+shtyp+"_AR "           # aa: after t_after and before t_after_after
                        query_aa+="where HARPNUM="+str(reg)+" "
                        query_aa+="and timestamp(tmax) between timestamp('"+t_after+"') and timestamp('"+t_after_after+"') "
                        query_aa+="and fflux>='"+str(cut0)+"' "
                        mycursor=mydb.cursor()
                        mycursor.execute(query_aa)
                        n_after_after=mycursor.fetchall()
                        naa=n_after_after[0][0]
                        C_aa=0
                        if naa>0:
                            C_aa=1
                    #
            ################################################################################# REGION LABEL
                        label="---"
                        if not C_b and not C_a and not C_aa:
                            label="NO1"
                        if not C_b and not C_a and     C_aa:
                            label="NO2"
                        if     C_b and not C_a and     C_aa:
                            label="NO3"
                        if     C_b and not C_a and not C_aa:
                            label="NO4"
                        if                 C_a:
                            label="YES"

            ################################################################################# 
                        start_obs=regions[regions["HARPNUM"]==reg]['start_obs'].iloc[0]
                        hours_staobs=regions[regions["HARPNUM"]==reg]['hours_staobs'].iloc[0]
                        end_obs=regions[regions["HARPNUM"]==reg]['end_obs'].iloc[0]
                        hours_endobs=regions[regions["HARPNUM"]==reg]['hours_endobs'].iloc[0]
                    
                        x=[[t_X,t_before,t_after,t_after_after,class0,format(cut0,".2e"),reg,label,C_b,C_a,C_aa,nb,na,naa,start_obs,hours_staobs,end_obs,hours_endobs]]
                        result=result._append(x)
                        print(t_X,t_before,t_after,t_after_after,class0,format(cut0,".2e"),reg,label," - ",C_b,C_a,C_aa," - ",nb,na,naa)
                
                    print("***************** Mind you have to drop the table every run *************")

                    engine=create_engine("mysql+mysqlconnector://stefano:stefano@localhost/stefano")
                    result.columns=['t_X','t_b','t_a','t_aa','class','flux_cut','HARPn','label','C_b','C_a','C_aa','n_b','n_a','n_aa','obs_start','obs_startH','obs_end','obs_endH']
        #################################################################
        ######################################################## OUTPUTS
        #################################################################
        #        result.to_sql(name="events_counts_2012-22", con=engine, if_exists="append", index=False)
        #################################################################
                if header_flag:
                    result.to_csv('DF_'+shtyp+'_'+class0+'_delta'+delta_hours+'_three_classes.csv',index=False,mode="a",header=True)
                    header_flag=False
                else:
                    result.to_csv('DF_'+shtyp+'_'+class0+'_delta'+delta_hours+'_three_classes.csv',index=False,mode="a",header=False)
                
        task.close()
        
    def create_hdf5_from_db(self, label_type, data_version="clean", table_type="DEF",
                            window_hours=24, buffer_minutes=6, sampling_interval=12,
                            test_mode=False, test_size=500, random_seed=42, db_type = "sqlite"):
        """
        Main public method: extract data from a database (SQLite or MySQL), match it
        with event/label information, generate time windows for each event, and save
        the resulting dataset into an HDF5 file.

        Parameters
        ----------
        label_type : str
            Type of label table to load (e.g. 'X', 'M', 'C').
        data_version : str, default "clean"
            Dataset version; determines which numeric features table to use.
        table_type : str, default "DEF"
            Table name prefix for feature tables.
        window_hours : int, default 24
            Backward time window size (hours) used to extract data before t_X.
        buffer_minutes : int, default 6
            Additional minutes before and after t_X to avoid boundary issues.
        sampling_interval : int, default 12
            Time rounding interval (in minutes) for T_OBS timestamps.
        test_mode : bool, default False
            Whether to randomly sample a small subset of events (for debugging).
        test_size : int, default 500
            Number of events to use when test_mode=True.
        random_seed : int, default 42
            Seed used when sub-sampling events in test mode.
        db_type : {"sqlite", "Mysql"}, default "sqlite"
            Selects which database backend to use.
        """
        print(f"\n[Cornerstone] Starting extraction from database...")

        task = _get_or_create_task(self, "DB to HDF5")
        # ----------------------------------------------------------------------
        # 1) Select database table and features based on configuration
        # ----------------------------------------------------------------------
        table_name, numeric_features = _select_db_config(self,data_version, table_type)
        # Categorical feature always included
        string_features = ["QUALITY"]
        features = numeric_features + string_features
        print(features) 
 
        # Construct label table name (e.g., DF_DEF_X_delta24_three_classes)    
        label_table_name = f"DF_{table_type}_{label_type}_delta{window_hours}_three_classes"
        
        # ======================================================================
        # 2) Load label/event table (different logic depending on DB backend)
        # ======================================================================     
        if db_type =="sqlite": 
            db_path = self.db_config["local_db_path"] 
            print(db_path)
            # Connect to SQLite and read the event/label table
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # dict-like access
            cursor = conn.cursor()

            try:
                query = f"""
                    SELECT *
                    FROM {label_table_name}
                    ORDER BY HARPn;
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description]
                 
                # Convert query results to DataFrame
                event_df = pd.DataFrame(rows, columns=colnames)
                event_df["t_X"] = pd.to_datetime(event_df["t_X"])

                total_events = len(event_df)
                print(f"\nTotal events to process: {total_events}")

                # If test mode: randomly sample only a small subset of events
                if test_mode:
                    print(f"Running in test mode with {test_size} events")
                    np.random.seed(random_seed)
                    event_df = event_df.sample(n=test_size, random_state=random_seed)
                    event_df = event_df.reset_index(drop=True)

            finally:
                cursor.close()
                conn.close()

            # Determine batch size
            batch_size = min(50 if test_mode else 1000, len(event_df))
            all_results, missing_data = [], []

            # ==================================================================
            # 3) Process events in batches
            # ==================================================================
            for i in range(0, len(event_df), batch_size):

                batch_df = event_df.iloc[i:i+batch_size]
                event_rows = batch_df.to_dict(orient="records")
                
                # Unique HARPNUMs in this batch
                harpnums = list(set(event["HARPn"] for event in event_rows))

                # Compute the time ranges for SQL lookup
                t_starts = [pd.to_datetime(e["t_X"]) - pd.Timedelta(hours=window_hours) - pd.Timedelta(minutes=buffer_minutes)
                            for e in event_rows]
                t_ends = [pd.to_datetime(e["t_X"]) + pd.Timedelta(minutes=buffer_minutes)
                        for e in event_rows]
                t_start_str = min(t_starts).strftime("%Y.%m.%d_%H:%M:%S_TAI")
                t_end_str = max(t_ends).strftime("%Y.%m.%d_%H:%M:%S_TAI")

                print(f"\n[Cornerstone] Batch {i//batch_size + 1}: querying {len(harpnums)} HARPNUMs")

                # --------------------------------------------------------------
                # Query SQLite features for this batch
                # --------------------------------------------------------------
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor() 
                try:
                    feature_str = ", ".join(features)
                    harp_list = ",".join(map(str, harpnums))

                    query = f"""
                        SELECT HARPNUM, T_OBS, {feature_str}
                        FROM {table_name}
                        WHERE HARPNUM IN ({harp_list})
                        AND T_OBS BETWEEN '{t_start_str}' AND '{t_end_str}'
                        ORDER BY HARPNUM, T_OBS;
                    """

                    cursor.execute(query)
                    data = cursor.fetchall()
                    # If no data exists for these events, register as missing
                    if not data:
                        for event in event_rows:
                            missing_data.append({
                                "HARPNUM": event["HARPn"],
                                "t_X": event["t_X"],
                                "label": event["label"],
                                "reason": "No data found in database for this batch"
                            })
                        continue

                    # Convert SQLite output to DataFrame
                    batch_data = pd.DataFrame(data, columns=["HARPNUM", "T_OBS"] + features)
                    batch_data["T_OBS"] = pd.to_datetime(
                        batch_data["T_OBS"],
                        format="%Y.%m.%d_%H:%M:%S_TAI"
                    ).dt.round(f"{sampling_interval}min")

                    # Process all events in this batch in parallel
                    batch_results, batch_missing = _process_batch_parallel(
                        self,
                        event_rows,
                        batch_data,
                        features,
                        numeric_features,
                        string_features,
                        window_hours,
                        buffer_minutes,
                        sampling_interval
                    )

                    # Accumulate results
                    all_results.extend(batch_results)
                    missing_data.extend(batch_missing)

                finally:
                    cursor.close()
                    conn.close()
        # ======================================================================
        # MySQL BACKEND
        # ======================================================================
        elif db_type == "Mysql":
            # Connect to MySQL and load label table
            mydb = mysql.connector.connect(**self.db_config)
            cursor = mydb.cursor(dictionary=True)
            try:
                query = f"""
                    SELECT *
                    FROM {label_table_name} 
                    ORDER BY HARPn ;
                """
                cursor.execute(query)
                #data = cursor.fetchall()
                event_df = pd.DataFrame(cursor.fetchall())
                event_df["t_X"] = pd.to_datetime(event_df["t_X"])
                total_events = len(event_df) 
                print(f"\nTotal events to process: {total_events}")

                # Select subset for testing if in test mode
                if test_mode:
                    print(f"Running in test mode with {test_size} events")
                    np.random.seed(random_seed)
                    event_df = event_df.sample(n=test_size, random_state=random_seed)
                    event_df = event_df.reset_index(drop=True)

            finally:    
                cursor.close()
                mydb.close()
            
            # Batch size logic identical to SQLite block
            batch_size = min(50 if test_mode else 1000, len(event_df))
            all_results, missing_data = [], []
            
            # Loop over event batches
            for i in range(0, len(event_df), batch_size):
                batch_df = event_df.iloc[i:i+batch_size]
                event_rows = batch_df.to_dict(orient="records")
                harpnums = list(set(event["HARPn"] for event in event_rows))

                # Compute time windows for SQL query
                t_starts = [pd.to_datetime(e["t_X"]) - pd.Timedelta(hours=window_hours) - pd.Timedelta(minutes=buffer_minutes)
                            for e in event_rows]
                t_ends = [pd.to_datetime(e["t_X"]) + pd.Timedelta(minutes=buffer_minutes) for e in event_rows]
                t_start_str = min(t_starts).strftime("%Y.%m.%d_%H:%M:%S_TAI")
                t_end_str = max(t_ends).strftime("%Y.%m.%d_%H:%M:%S_TAI")

                print(f"\n[Cornerstone] Batch {i//batch_size + 1}: querying {len(harpnums)} HARPNUMs")

                # Query MySQL table for this batch
                mydb = mysql.connector.connect(**self.db_config)
                cursor = mydb.cursor()
                try:
                    query = f"""
                        SELECT HARPNUM, T_OBS, {", ".join(features)}
                        FROM {table_name}
                        WHERE HARPNUM IN ({','.join(map(str, harpnums))})
                        AND T_OBS BETWEEN '{t_start_str}' AND '{t_end_str}'
                        ORDER BY HARPNUM, T_OBS;
                    """
                    cursor.execute(query)
                    data = cursor.fetchall()
                    if not data:
                        # Register missing events
                        for event in event_rows:
                            missing_data.append({
                                "HARPNUM": event["HARPn"],
                                "t_X": event["t_X"],
                                "label": event["label"],
                                "reason": "No data found in database for this batch"
                            })
                        continue

                    # Convert MySQL output to DataFrame
                    batch_data = pd.DataFrame(data, columns=["HARPNUM", "T_OBS"] + features)
                    batch_data["T_OBS"] = pd.to_datetime(batch_data["T_OBS"],
                                                        format="%Y.%m.%d_%H:%M:%S_TAI").dt.round(f"{sampling_interval}min")

                    # Process in parallel
                    batch_results, batch_missing = _process_batch_parallel(
                        self, event_rows, batch_data, features, numeric_features,
                        string_features, window_hours, buffer_minutes, sampling_interval
                    )

                    all_results.extend(batch_results)
                    missing_data.extend(batch_missing)

                finally:
                    cursor.close()
                    mydb.close()

        # ======================================================================
        # 4) SAVE HDF5 OUTPUT
        # ======================================================================
        save_path = os.path.join(
            self.output_folder,
            f"sharp_dataset_db_{table_name}_{label_type}_w{window_hours}h_s{sampling_interval}min.h5"
        ) 
        print(f"saving sharp_dataset_db_{table_name}_{label_type}_w{window_hours}h_s{sampling_interval}min.h5 into {save_path}")
        _save_hdf5(
            self,
            save_path,
            all_results,
            numeric_features,
            data_version,
            table_name,
            str(label_type) + "_delta" + str(window_hours),
            all_results
        )
        
        # Save missing-event log
        _save_missing_log(self , missing_data)

        task.close()
        # Final reporting
        print(f"\n[Cornerstone] Extraction complete! Saved to: {save_path}")
        print(f"Processed {len(all_results)} / {total_events} events")
  
    def create_dataset_splits(self, config):
        """
        Main public method: preprocess the dataset and create train/validation/test splits.
        
        Parameters:
        -----------
        config : dict or class
            Configuration parameters for preprocessing, filtering, feature selection, and splitting.
            Must include the same attributes as expected by the pipeline script.
        
        Returns:
        --------
        None
            Saves preprocessed splits and config to disk; prints logs.
        """
        
        # --- Step 0: Print header and create logging task ---
        print("SHARP Dataset Preprocessing and Splitting Pipeline")
        print("=" * 60)
        print()
        task = _get_or_create_task(self, "Dataset Split")  # Create a logging/task tracking object
        
        # --- Step 1: Convert config dict/class into SimpleNamespace for easy attribute access ---
        config = SimpleNamespace(**config)

        # --- Step 2: Print configuration summary ---
        _print_config_summary(self, config)
        print()

        # --- Step 3: Generate a unique experiment name based on the config ---
        experiment_name = _generate_experiment_name(self, config)
        print(f"Experiment Name: {experiment_name}")
        print()

        # --- Step 4: Create output directories for splits, plots, and config ---
        output_dir = _create_output_directories(self, config, experiment_name)
        print(f"Output directory: {output_dir}")
        print()

        # --- Step 5: Save the configuration to disk for reproducibility ---
        _save_config(self, config, output_dir)
        print("Configuration saved")
        print()

        # --- Step 6: Load dataset from HDF5 using SharpDataset wrapper ---
        dataset = SharpDataset(
            h5_path=config.INPUT_H5_FILE,
            timesteps=config.TIMESTEPS,
            label=config.LABEL_TYPE,
            use_mask=config.USE_MASK,
            multi_class=config.MULTI_CLASS
        )
        print(f"Dataset loaded: {len(dataset)} samples, {len(dataset.feature_names)} features")
        dataset.overview() # Print feature overview and dataset stats
        print()

        # --- Step 7: Feature selection ---
        if config.FEATURE_SELECTION_MODE == "criterion":
            # Select features based on a criterion (e.g., importance, correlation)
            selector = FeatureSelector(criterion=config.FEATURE_CRITERION)
        else:
            # Directly use the features specified in config
            selector = FeatureSelector(selected_features=config.DIRECT_FEATURES)
        dataset = selector(dataset) # Apply feature selection
        print(f"Feature selection complete: {len(selector.dataset_before if hasattr(selector,'dataset_before') else dataset.feature_names)} -> {len(dataset.feature_names)}")
        print()

        # --- Step 8: Missingness filtering ---
        if config.MISSINGNESS_FILTER_MODE == "single_window":
            # Filter samples with NaNs above max threshold in a single window
            filter_transform = MissingnessFilter(max_nan_ratio=config.MAX_NAN_RATIO)
            dataset = filter_transform(dataset)
        elif config.MISSINGNESS_FILTER_MODE == "multi_window":
            # Filter samples with NaNs above threshold in multiple time windows
            filter_transform = MultiWindowMissingnessFilter(time_windows=config.TIME_WINDOWS,
                                                           max_nan_ratio=config.MAX_NAN_RATIO,
                                                           verbose=config.VERBOSE)
            dataset = filter_transform(dataset)
            if config.VERBOSE:
                summary = filter_transform.get_filtering_summary()
                print(f"Filtering summary: {summary}")
        elif config.MISSINGNESS_FILTER_MODE == "multi_window_empty":
            # Filter samples empty across multiple windows
            filter_transform = MultiWindowEmptyFilter(time_windows=config.TIME_WINDOWS, verbose=config.VERBOSE)
            dataset = filter_transform(dataset)
            if config.VERBOSE:
                summary = filter_transform.get_filtering_summary()
                print(f"Filtering summary: {summary}")
        else:
            print("No missingness filtering applied")
        print(f"After missingness filtering: {len(dataset)} samples")
        print()

        # --- Step 9: Temporal filtering ---
        if config.TEMPORAL_FILTER_MODE is not None and config.TEMPORAL_WINDOW is not None:
            start_date, end_date = config.TEMPORAL_WINDOW
            if config.TEMPORAL_FILTER_MODE == "select":
                # Keep only samples within the time window
                tf = TimeWindowSelector(start_date=start_date, end_date=end_date, verbose=config.VERBOSE)
                dataset = tf(dataset)
            elif config.TEMPORAL_FILTER_MODE == "filter":
                # Keep only samples within the time window
                tf = TimeWindowFilter(start_date=start_date, end_date=end_date, verbose=config.VERBOSE)
                dataset = tf(dataset)
            elif config.TEMPORAL_FILTER_MODE == "exclude":
                # Exclude samples within the time window
                tf = TimeWindowExcluder(start_date=start_date, end_date=end_date, verbose=config.VERBOSE)
                dataset = tf(dataset)
            else:
                print(f"Unknown temporal filter mode: {config.TEMPORAL_FILTER_MODE}")
            print(f"After temporal filtering: {len(dataset)} samples")
            print()
        else:
            print("No temporal filtering applied")
            print()

        # --- Step 10: Multi-period filtering ---
        if config.MULTI_PERIOD_FILTER and config.PROBLEMATIC_PERIODS:
            filters = []
            for start_date, end_date in config.PROBLEMATIC_PERIODS:
                # Create filter for each problematic period
                filters.append(TimeWindowFilter(start_date=start_date, end_date=end_date, verbose=False))
            multi_period_filter = Compose(filters)
            dataset = multi_period_filter(dataset)
            print(f"After multi-period filtering: {len(dataset)} samples (excluded {len(config.PROBLEMATIC_PERIODS)} periods)")
            print()
        else:
            print("No multi-period filtering applied")
            print()

        # --- Step 11: CMASK filtering ---
        if config.CMASK_FILTER:
            cm = CMASKFilter(threshold=config.CMASK_THRESHOLD,
                             strategy=config.CMASK_STRATEGY,
                             majority_threshold=config.CMASK_MAJORITY_THRESHOLD,
                             verbose=config.VERBOSE)
            dataset = cm(dataset)  # Apply CMASK filtering
            if config.VERBOSE:
                summary = cm.get_filtering_summary()
                print(f"CMASK filtering summary: {summary}")
            print(f"After CMASK filtering: {len(dataset)} samples")
            print()
        else:
            print("No CMASK filtering applied")
            print()

        # --- Step 12: Create dataset splits ---
        print("Creating dataset splits...")
        splitter = SharpDatasetSplitter(
            strategy=config.SPLITTING_STRATEGY,
            test_size=config.TEST_SIZE,
            n_folds=config.N_FOLDS,
            random_state=config.RANDOM_STATE,
            temporal_distr_viz=config.SAVE_PLOTS,
            label_distr_viz=config.SAVE_PLOTS,
            output_dir=str(Path(output_dir) / "plots"),
            force_separation=config.FORCE_SEPARATION
        )

        splits = splitter.split(
            dataset=dataset,
            test_time_window=getattr(config, "TEST_TIME_WINDOW", None),
            test_cutoff_date=getattr(config, "TEST_CUTOFF_DATE", None),
            verbose=config.VERBOSE
        )

        print(f"Splits created successfully!")
        print(f"  - Train folds: {len(splits['train'])}")
        print(f"  - Val folds: {len(splits['val'])}")
        print(f"  - Test sets: {len(splits['test'])}")
        print()

        # --- Step 13: Save splits to HDF5 ---
        splits_dir = Path(output_dir) / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        # save test split
        test_subset = splits["test"][0]
        test_path = splits_dir / f"{experiment_name}_test.h5"
        save_split_to_h5(test_subset, str(test_path), verbose=config.VERBOSE)

        # save train/validation folds
        for fold in range(len(splits["train"])):
            train_subset = splits["train"][fold]
            train_path = splits_dir / f"{experiment_name}_cv_{fold}_train.h5"
            save_split_to_h5(train_subset, str(train_path), verbose=config.VERBOSE)

            val_subset = splits["val"][fold]
            val_path = splits_dir / f"{experiment_name}_cv_{fold}_val.h5"
            save_split_to_h5(val_subset, str(val_path), verbose=config.VERBOSE)

        task.close()
        # --- Step 14: Final logging ---
        print()
        print("Pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print()
        print("Available files:")
        print(f"  - Splits: {output_dir}/splits/")
        print(f"  - Plots: {output_dir}/plots/")
        print(f"  - Config: {output_dir}/config/") 
 
    def train_models(
        self , train_config_path=None
    ):
        """
        High-level wrapper to run cross-validation training for multiple models.
        This mirrors the workflow of flare_forecasting/main.py.

        Parameters
        ----------
        train_config_path : str, optional
            Path to a custom training configuration file (.py).
            If provided, it overrides the default config module.

        Workflow
        --------
        1. Load configuration (default or custom)
        2. For each model in CURRENT_MODELS:
            - Initialize managers (paths, configs, metrics)
            - Save configuration files for reproducibility
            - Run cross-validation training
            - Save metrics and plots
        """
        from .ml.utils.config import get_training_config 
        from .ml.utils import config as cfg
        
        # Create and register task in task manager system
        task = _get_or_create_task( self, "Train Model")
         
        # ----------------------------------------------------
        # 1. Load configuration (default or user-specified)
        # ----------------------------------------------------
        if train_config_path is not None:  
            # Dynamically override config module path
            cfg.set_config_path(train_config_path)
        #cfg = _load_config("cornerstone/ml/utils/config.py")

        # Dynamically override config module path
        MODEL_CONFIGS = cfg.MODEL_CONFIGS
        DATA_CONFIG = cfg.DATA_CONFIG
        DEVICE_CONFIG = cfg.DEVICE_CONFIG
        OUTPUT_CONFIG = cfg.OUTPUT_CONFIG
        TASK_CONFIG = cfg.TASK_CONFIG
        CURRENT_DATASET = cfg.CURRENT_DATASET
        CURRENT_MODELS = cfg.CURRENT_MODELS
         
        # Dynamically override config module path
        self.tasks["Train Model"].connect(CURRENT_DATASET(), f"dataset configs")
        self.tasks["Train Model"].connect(DATA_CONFIG(), f"data configs")
        
        # ----------------------------------------------------
        # 2. Train models listed in CURRENT_MODELS
        # ----------------------------------------------------
        for CURRENT_MODEL in CURRENT_MODELS():
            print(f"\n=== Starting training for model: {CURRENT_MODEL} ===")
            
            # Model-specific configuration
            MODEL_CONFIG = MODEL_CONFIGS.get(CURRENT_MODEL, {})
            TRAINING_CONFIG = get_training_config(CURRENT_MODEL)
 
            # Log configs in ClearML
            self.tasks["Train Model"].connect(TRAINING_CONFIG, f"training config {CURRENT_MODEL}")
            self.tasks["Train Model"].connect(MODEL_CONFIG, f"configs {CURRENT_MODEL}")
 
            # Retrieve global task configs
            task_config = TASK_CONFIG()
            output_config = OUTPUT_CONFIG()
            
            
            # Used when running ablation studies (none here)
            run_name = None 
            
            # ------------------------------------------------
            # 3. Initialize managers for paths, models, metrics
            # ------------------------------------------------
            path_manager = PathManager(
                task_name=task_config["name"],
                model_type=CURRENT_MODEL,
                dataset_name=task_config["data_prefix"],
                base_dir=output_config["results_dir"],
                run_name=run_name
            )
            model_manager = ModelManager(CURRENT_MODEL)#ModelManager()
            metrics_manager = MetricsManager(path_manager)
            config_manager = ConfigManager(path_manager)

            # Print directory info (useful for debugging)
            print(f"Results will be saved to: {path_manager.get_results_structure()}")
            print(f"Dataset: {path_manager.get_dataset_name()}")
            print(f"Task: {path_manager.get_task_name()}")
            print(f"Model: {path_manager.get_model_type()}")

            # Ensure directory layout exists
            path_manager.ensure_dir("models")
            path_manager.ensure_dir("plots")
            path_manager.ensure_dir("metrics")
            path_manager.ensure_dir("attention_maps")
            path_manager.ensure_dir("configs")

            # ------------------------------------------------
            # 4. Save config for reproducibility
            # ------------------------------------------------
            print("\n" + config_manager.get_config_summary(
                current_dataset=CURRENT_DATASET,
                current_model=CURRENT_MODEL,
                data_config=DATA_CONFIG
            ))

            # Save config (either using run_name or default naming)
            if run_name:
                config_filename = f"{run_name}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                json_filename = f"{run_name}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                config_manager.save_config(
                    CURRENT_MODEL,
                    filename=config_filename,
                    config_obj=cfg
                )   
                config_manager.save_config_json(
                    CURRENT_MODEL,
                    filename=json_filename,
                    config_obj=cfg
                )   
            else:
                config_manager.save_config(CURRENT_MODEL, config_obj=cfg)  
                config_manager.save_config_json(CURRENT_MODEL, config_obj=cfg)  

            # ------------------------------------------------
            # 5. Run cross-validation training for THIS model
            # ------------------------------------------------
            summary_metrics = run_cv_training(
                model_name=CURRENT_MODEL,
                window_map=DATA_CONFIG()["window_map"],
                base_path=DATA_CONFIG()["base_path"],
                model_fn=lambda steps: model_manager.create_model(CURRENT_MODEL,**{**MODEL_CONFIG,
                                                                                   **({"seq_len": steps} if "seq_len" in MODEL_CONFIG else {})}),
                patience=TRAINING_CONFIG["patience"],
                num_epochs=TRAINING_CONFIG["num_epochs"],
                n_folds=TRAINING_CONFIG["n_folds"],
                device=DEVICE_CONFIG["device"],
                batch_size=TRAINING_CONFIG["batch_size"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                output_config=output_config,
                path_manager=path_manager,
                metrics_file=path_manager.get_metrics_path(f"{task_config['name']}_metrics.json"),
                task_config=task_config,
                lr_scheduler_config=TRAINING_CONFIG["lr_scheduler"],
                loss_config=TRAINING_CONFIG["loss_function"],
                threshold_config=TRAINING_CONFIG["threshold_optimization"],
                calibration_config=TRAINING_CONFIG.get("calibration", None),
                early_stopping_config=TRAINING_CONFIG.get("early_stopping", None),
                clearml_task=self.tasks["Train Model"],
                regularization_params = TRAINING_CONFIG.get("regularization_params",None)
            )
            task.close() 

    def test_model(self, test_config_path=None):
        """
        High-level wrapper to run model testing on the SHARP dataset.
        This function mirrors the testing workflow of flare_forecasting/main.py.

        Parameters
        ----------
        test_config_path : str, optional
            Path to a custom .py config file.
            If provided, it overrides the default configuration module.

        Workflow
        --------
        1. Load configuration (default or overridden)
        2. For each model listed in CURRENT_MODELS:
            - Initialize path, config, and metrics managers
            - Ensure directory structure exists
            - Save configuration for reproducibility
            - Load model and run test evaluation
        """

        # ---------------------------------------------------------
        # 1. Load configuration (default or custom user-provided)
        # ---------------------------------------------------------
        from .ml.utils.config import get_training_config
        from .ml.utils import config as cfg 
        
        # Register task for monitoring (ClearML or internal system)
        task = _get_or_create_task(self, "Test Model")   
        
        # If user provided a custom config, override default module
        if test_config_path is not None:  
            cfg.set_config_path(test_config_path)
        
        # Extract relevant config sections — these remain consistent
        MODEL_CONFIGS = cfg.MODEL_CONFIGS 
        DATA_CONFIG = cfg.DATA_CONFIG
        DEVICE_CONFIG = cfg.DEVICE_CONFIG
        OUTPUT_CONFIG = cfg.OUTPUT_CONFIG
        TASK_CONFIG = cfg.TASK_CONFIG
        CURRENT_DATASET = cfg.CURRENT_DATASET
        CURRENT_MODELS = cfg.CURRENT_MODELS

        # Log relevant configs
        self.tasks["Test Model"].connect(CURRENT_DATASET(), f"dataset configs") 
        self.tasks["Test Model"].connect(DATA_CONFIG(), f"data configs")
        
        # ---------------------------------------------------------
        # 2. Iterate through each model to be tested
        # ---------------------------------------------------------
        for CURRENT_MODEL in CURRENT_MODELS():
            print(f"\n=== Starting testing for model: {CURRENT_MODEL} ===")
            
            # Retrieve model-specific configuration
            MODEL_CONFIG = MODEL_CONFIGS.get(CURRENT_MODEL, {})  
            TRAINING_CONFIG = get_training_config(CURRENT_MODEL)
 
            # Log training config and model config for reproducibility
            self.tasks["Test Model"].connect(TRAINING_CONFIG, f"training config {CURRENT_MODEL}")
            self.tasks["Test Model"].connect(MODEL_CONFIG, f"configs {CURRENT_MODEL}")

            # Placeholder: run name (for ablation studies, not used here)
            run_name = None 
            
            # ---------------------------------------------------------
            # 3. Initialize managers for paths, configs, and metrics
            # ---------------------------------------------------------
            path_manager = PathManager(
                task_name=TASK_CONFIG()["name"],
                model_type=CURRENT_MODEL,
                dataset_name=TASK_CONFIG()["data_prefix"],
                base_dir=OUTPUT_CONFIG()["results_dir"],
                run_name=run_name
            )
            model_manager = ModelManager(CURRENT_MODEL)#ModelManager()
            metrics_manager = MetricsManager(path_manager)
            config_manager = ConfigManager(path_manager)

            # Display filesystem structure for transparency/debugging
            print(f"Results will be saved to: {path_manager.get_results_structure()}")
            print(f"Dataset: {path_manager.get_dataset_name()}")
            print(f"Task: {path_manager.get_task_name()}")
            print(f"Model: {path_manager.get_model_type()}")

            # Ensure required directory structure exists
            path_manager.ensure_dir("models")
            path_manager.ensure_dir("plots")
            path_manager.ensure_dir("metrics")
            path_manager.ensure_dir("attention_maps")
            path_manager.ensure_dir("configs")

            # ---------------------------------------------------------
            # 4. Save configuration files (text + JSON) for reproducibility
            # ---------------------------------------------------------
            print("\n" + config_manager.get_config_summary(
                current_dataset=CURRENT_DATASET,
                current_model=CURRENT_MODEL,
                data_config=DATA_CONFIG
            ))
            
            if run_name:
                # Special case for ablation runs (not used here)
                config_filename = f"{run_name}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                json_filename = f"{run_name}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                config_manager.save_config(
                    CURRENT_MODEL,
                    filename=config_filename,
                    config_obj=cfg
                )  # Save as readable text file
                config_manager.save_config_json(
                    CURRENT_MODEL,
                    filename=json_filename,
                    config_obj=cfg
                )  # Save as JSON for programmatic access
            else:
                # Standard save format
                config_manager.save_config(CURRENT_MODEL, config_obj=cfg)  # Save as readable text file
                config_manager.save_config_json(CURRENT_MODEL, config_obj=cfg)  # Save as JSON for programmatic access
 
            # ---------------------------------------------------------
            # 5. Run final test set evaluation
            # ---------------------------------------------------------
            run_test(
                model_name=CURRENT_MODEL,
                window_map=DATA_CONFIG()["window_map"],  
                base_path=DATA_CONFIG()["base_path"],
                data_prefix=TASK_CONFIG()["data_prefix"], 
                model_checkpoint_path=None, #"/bigdisk/daniele/cornerstone_module/results/sharp_dataset_db_SHARPC_clean_NRT_direct_17features_multiwin_empty_win1-16-31-61-121_no_cmask_M+_random_20pct/M+/models/M+_minirocket_model_3h_fold4.pkl",
                clearml_task=self.tasks["Test Model"],
                path_manager= path_manager, 
                model_fn=lambda steps: model_manager.create_model(CURRENT_MODEL,**{**MODEL_CONFIG,
                                                                                   **({"seq_len": steps} if "seq_len" in MODEL_CONFIG else {})})                 
            )
            task.close()


def _select_db_config(self, data_version, table_type):
    """Return table name and feature list depending on version."""
    if data_version == "clean":
        numeric_features = [
            "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX_N1", "MEANGBT",
            "MEANGBZ_N2", "MEANGBH_N2", "MEANJZD", "TOTUSJZ_N1", "MEANJZH",
            "TOTUSJH_N1", "ABSNJZH", "MEANPOT", "SHRGT45", "R_VALUE", "CMASK"
        ]
    else:
        numeric_features = [
            "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX", "MEANGAM",
            "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "TOTUSJZ", "MEANALP",
            "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT",
            "MEANSHR", "SHRGT45", "R_VALUE", "CMASK"
        ]
    table_name = f"SHARPC_{data_version}_{table_type}"
    return table_name, numeric_features

def _save_hdf5(self, path, results, numeric_features, data_version, table_name, label_suffix, all_results):
    # Extract arrays from the result dictionaries
    X_data = np.array([r["X"] for r in results])
    Y_data = np.array([r["Y"] for r in results], dtype="S")  # save strings as bytes
    harp_nums = np.array([r["HARPNUM"] for r in results])
    timestamps = np.array([r["timestamps"] for r in results], dtype="S")

    # Extract metadata fields
    quality_values = [r["string_metadata"]["QUALITY"] for r in results]

    # Labels (from all_results)
    label_u_data = np.array([r["label_u"] for r in all_results], dtype="S")
    label_data = np.array([r["label"] for r in all_results], dtype="S")

    # Open HDF5 file for writing
    with h5py.File(path, "w") as hf:
        # Store datasets
        hf.create_dataset("X", data=X_data, compression="gzip")
        hf.create_dataset("Y", data=Y_data)
        hf.create_dataset("HARPNUM", data=harp_nums)
        hf.create_dataset("timestamps", data=timestamps)

        # Store labels (with optional suffix)
        if label_suffix:
            hf.create_dataset(f"label_{label_suffix}", data=label_data)
            hf.create_dataset(f"label_u_{label_suffix}", data=label_u_data)
        else:
            hf.create_dataset("label", data=label_data)
            hf.create_dataset("label_u", data=label_u_data)

        # Add metadata as HDF5 attributes
        hf.attrs["feature_names"] = ",".join(numeric_features)
        hf.attrs["database_version"] = data_version
        hf.attrs["table_name"] = table_name
        hf.attrs["quality"] = ",".join(quality_values)
        
def _save_missing_log(self, missing_data):
    if not missing_data:
        return
    path = os.path.join(self.output_folder, "missing_data_log.csv")
    pd.DataFrame(missing_data).to_csv(path, index=False)
    print(f"[Cornerstone] Missing data log saved → {path}")
    
    
    
# ==============================
# --- PARALLEL PROCESSING ---
# ============================== 
def _process_events_parallel(args):
    """Process multiple events in parallel, using pre-fetched data."""
    import numpy as np
    import pandas as pd

    events, batch_data, features, numeric_features, string_features, window_hours, buffer_minutes, sampling_interval = args

    results = []
    missing_events = []

    for event in events:
        harpnum = event["HARPn"]
        t_X = pd.to_datetime(event["t_X"])
        t_start = t_X - pd.Timedelta(hours=window_hours)

        event_df = batch_data[batch_data["HARPNUM"] == harpnum].copy()
        event_df = event_df[(event_df["T_OBS"] >= t_start) & (event_df["T_OBS"] <= t_X)]
        event_df = event_df.drop_duplicates(subset=["T_OBS"])

        if event_df.empty:
            harpnum_exists = not batch_data[batch_data["HARPNUM"] == harpnum].empty
            reason = "No data found in time window" if harpnum_exists else "HARPNUM not found in database"
            missing_events.append({
                "HARPNUM": harpnum,
                "t_X": t_X,
                "label": event["label"],
                "reason": reason
            })
            continue

        full_index = pd.date_range(start=t_start, end=t_X, freq=f"{sampling_interval}min")
        event_df.set_index("T_OBS", inplace=True)
        event_df = event_df.reindex(full_index)

        X = event_df[numeric_features].to_numpy(dtype=np.float32)
        timestamps = full_index.astype(str).to_numpy()

        string_metadata = {}
        for col in string_features:
            if col in event_df.columns and not event_df[col].isna().all():
                vc = event_df[col].value_counts()
                string_metadata[col] = str(vc.index[0]) if not vc.empty else "NULL"
            else:
                string_metadata[col] = "NULL"

        results.append({
            "X": X,
            "string_metadata": string_metadata,
            "Y": event["label"],
            "HARPNUM": harpnum,
            "timestamps": timestamps,
            "label_u": event["label_u"],
            "label": event["label"]
        })

    return results, missing_events

def _process_batch_parallel(self, event_rows, batch_data, features, numeric_features,
                            string_features, window_hours, buffer_minutes, sampling_interval):
    """Process a batch of events using multiple CPU cores."""
    num_cores = cpu_count()
    total_events = len(event_rows)
    base_chunk_size = total_events // num_cores
    remainder = total_events % num_cores

    event_chunks, start_idx = [], 0
    for i in range(num_cores):
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        if chunk_size > 0:
            event_chunks.append(event_rows[start_idx:start_idx + chunk_size])
            start_idx += chunk_size

    chunk_args = []
    for chunk in event_chunks:
        chunk_harpnums = {event["HARPn"] for event in chunk}
        chunk_data = batch_data[batch_data["HARPNUM"].isin(chunk_harpnums)].copy()
        chunk_args.append((
            chunk,
            chunk_data,
            features,
            numeric_features,
            string_features,
            window_hours,
            buffer_minutes,
            sampling_interval
        ))

    with Pool(processes=num_cores) as pool:
        chunk_results = list(tqdm(
            pool.imap_unordered(_process_events_parallel, chunk_args),
            total=len(chunk_args)
        ))

    all_results, all_missing = [], []
    for results, missing in chunk_results:
        if results:
            all_results.extend(results)
        if missing:
            all_missing.extend(missing)

    return all_results, all_missing



def _generate_experiment_name(self, config):
    """Generate a descriptive experiment name based on configuration."""
    if getattr(config, "EXPERIMENT_NAME", None):
        return config.EXPERIMENT_NAME

    input_filename = os.path.basename(config.INPUT_H5_FILE)
    if "w24h_s12min_multilabel" in input_filename:
        dataset_source = input_filename.split("w24h_s12min_multilabel")[0]
        if dataset_source.endswith("_"):
            dataset_source = dataset_source[:-1]
    else:
        dataset_source = os.path.splitext(input_filename)[0]

    if config.FEATURE_SELECTION_MODE == "criterion":
        feature_name = config.FEATURE_CRITERION
    else:
        feature_name = f"direct_{len(config.DIRECT_FEATURES)}features"

    if config.MISSINGNESS_FILTER_MODE == "multi_window":
        nan_info = f"multiwin{int(config.MAX_NAN_RATIO*100)}"
        window_info = f"win{'-'.join(map(str, config.TIME_WINDOWS))}"
        nan_info = f"{nan_info}_{window_info}"
    elif config.MISSINGNESS_FILTER_MODE == "multi_window_empty":
        nan_info = "multiwin_empty"
        window_info = f"win{'-'.join(map(str, config.TIME_WINDOWS))}"
        nan_info = f"{nan_info}_{window_info}"
    elif config.MISSINGNESS_FILTER_MODE == "single_window":
        nan_info = f"nan{int(config.MAX_NAN_RATIO*100)}"
    else:
        nan_info = "no_filter"

    if config.CMASK_FILTER:
        cmask_info = f"cmask{config.CMASK_STRATEGY}{config.CMASK_THRESHOLD}"
    else:
        cmask_info = "no_cmask"

    label_type = config.LABEL_TYPE
    if label_type == "Mplus_delta24":
        label_info = "M+"
    elif label_type == "Cplus_delta24":
        label_info = "C+"
    elif label_type == "Xplus_delta24":
        label_info = "X+"
    else:
        label_info = label_type.replace("_delta24", "")

    if config.SPLITTING_STRATEGY == "chronological":
        if getattr(config, "TEST_CUTOFF_DATE", None):
            split_info = f"chrono_cutoff_{config.TEST_CUTOFF_DATE}"
        elif getattr(config, "TEST_TIME_WINDOW", None):
            start, end = config.TEST_TIME_WINDOW
            split_info = f"chrono_window_{start}_to_{end}"
        else:
            split_info = f"chrono_default_{int(config.TEST_SIZE*100)}pct"
    else:
        split_info = f"random_{int(config.TEST_SIZE*100)}pct"

    return f"{dataset_source}_{feature_name}_{nan_info}_{cmask_info}_{label_info}_{split_info}"

def _create_output_directories(self, config, experiment_name):
    """Create output directory structure."""
    output_dir = Path(config.OUTPUT_BASE_DIR) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "splits").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    return output_dir

def _save_config(self, config, output_dir):
    """Save configuration to a file for reproducibility."""
    config_file = Path(output_dir) / "config" / "config.txt"
    with open(config_file, "w") as f:
        f.write("Dataset Preprocessing and Splitting Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write("Experiment Name: {}\n".format(_generate_experiment_name(self, config)))
        f.write("Created: {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Input/Output Settings:\n")
        f.write("  Input H5 File: {}\n".format(config.INPUT_H5_FILE))
        f.write("  Output Directory: {}\n\n".format(output_dir))
        f.write("Feature Selection Settings:\n")
        f.write("  Mode: {}\n".format(config.FEATURE_SELECTION_MODE))
        if config.FEATURE_SELECTION_MODE == "criterion":
            f.write("  Criterion: {}\n".format(config.FEATURE_CRITERION))
        else:
            f.write("  Direct Features: {}\n".format(config.DIRECT_FEATURES))
        f.write("\n")
        f.write("Missingness Filtering Settings:\n")
        f.write("  Mode: {}\n".format(config.MISSINGNESS_FILTER_MODE))
        if config.MISSINGNESS_FILTER_MODE == "multi_window_empty":
            f.write("  Max NaN Ratio: N/A (empty sample filtering only)\n")
        else:
            f.write("  Max NaN Ratio: {}\n".format(config.MAX_NAN_RATIO))
        if config.MISSINGNESS_FILTER_MODE in ["multi_window", "multi_window_empty"]:
            f.write("  Time Windows: {}\n".format(config.TIME_WINDOWS))
        f.write("\n")
        f.write("Temporal Filtering Settings:\n")
        f.write("  Mode: {}\n".format(config.TEMPORAL_FILTER_MODE))
        if getattr(config, "TEMPORAL_WINDOW", None):
            f.write("  Temporal Window: {}\n".format(config.TEMPORAL_WINDOW))
        f.write("\n")
        f.write("Multi-Period Filtering Settings:\n")
        f.write("  Multi-Period Filter: {}\n".format(config.MULTI_PERIOD_FILTER))
        if getattr(config, "PROBLEMATIC_PERIODS", None):
            f.write("  Problematic Periods: {}\n".format(config.PROBLEMATIC_PERIODS))
        f.write("CMASK Filtering Settings:\n")
        f.write("  CMASK Filter: {}\n".format(config.CMASK_FILTER))
        f.write("  CMASK Threshold: {}\n".format(config.CMASK_THRESHOLD))
        f.write("  CMASK Strategy: {}\n".format(config.CMASK_STRATEGY))
        f.write("  CMASK Majority Threshold: {}\n".format(config.CMASK_MAJORITY_THRESHOLD))
        f.write("Dataset Settings:\n")
        f.write("  Timesteps: {}\n".format(config.TIMESTEPS))
        f.write("  Label Type: {}\n".format(config.LABEL_TYPE))
        f.write("  Use Mask: {}\n".format(config.USE_MASK))
        f.write("  Multi Class: {}\n\n".format(config.MULTI_CLASS))
        f.write("Splitting Settings:\n")
        f.write("  Strategy: {}\n".format(config.SPLITTING_STRATEGY))
        f.write("  Test Size: {} ({:.0f}%)\n".format(config.TEST_SIZE, config.TEST_SIZE*100))
        f.write("  N Folds: {}\n".format(config.N_FOLDS))
        f.write("  Random State: {}\n".format(config.RANDOM_STATE))
        if getattr(config, "TEST_CUTOFF_DATE", None):
            f.write("  Test Cutoff Date: {}\n".format(config.TEST_CUTOFF_DATE))
        if getattr(config, "TEST_TIME_WINDOW", None):
            f.write("  Test Time Window: {}\n".format(config.TEST_TIME_WINDOW))
        f.write("  Force Separation: {}\n\n".format(config.FORCE_SEPARATION))
        f.write("Output Settings:\n")
        f.write("  Save Plots: {}\n".format(config.SAVE_PLOTS))
        f.write("  Verbose: {}\n".format(config.VERBOSE))

def _print_config_summary(self, config):
    """Print a short summary, same as your script."""
    print("Configuration Summary")
    print("=" * 50)
    print(f"Input: {config.INPUT_H5_FILE}")
    print(f"Output: {config.OUTPUT_BASE_DIR}")
    print()
    print("Feature Selection:")
    if config.FEATURE_SELECTION_MODE == "criterion":
        print(f"  Mode: Criterion-based ({config.FEATURE_CRITERION})")
    else:
        print(f"  Mode: Direct selection ({len(config.DIRECT_FEATURES)} features)")
    print()
    print("Missingness Filtering:")
    print(f"  Mode: {config.MISSINGNESS_FILTER_MODE}")
    if config.MISSINGNESS_FILTER_MODE == "multi_window_empty":
        print(f"  Max NaN Ratio: N/A (empty sample filtering only)")
    else:
        print(f"  Max NaN Ratio: {config.MAX_NAN_RATIO}")
    if config.MISSINGNESS_FILTER_MODE in ["multi_window", "multi_window_empty"]:
        print(f"  Time Windows: {config.TIME_WINDOWS}")
    print()
    print("Temporal Filtering:")
    print(f"  Mode: {config.TEMPORAL_FILTER_MODE}")
    if getattr(config, "TEMPORAL_WINDOW", None):
        print(f"  Temporal Window: {config.TEMPORAL_WINDOW}")
    print()
    print("Multi-Period Filtering:")
    print(f"  Multi-Period Filter: {config.MULTI_PERIOD_FILTER}")
    if getattr(config, "PROBLEMATIC_PERIODS", None):
        print(f"  Problematic Periods: {config.PROBLEMATIC_PERIODS}")
    print()
    print("CMASK Filtering:")
    print(f"  CMASK Filter: {config.CMASK_FILTER}")
    print(f"  CMASK Threshold: {config.CMASK_THRESHOLD}")
    print(f"  CMASK Strategy: {config.CMASK_STRATEGY}")
    print(f"  CMASK Majority Threshold: {config.CMASK_MAJORITY_THRESHOLD}")
    print()
    print("Splitting Strategy:")
    print(f"  Method: {config.SPLITTING_STRATEGY}")
    print(f"  Test Size: {config.TEST_SIZE*100:.0f}%")
    print(f"  Cross-validation: {config.N_FOLDS} folds")
    if getattr(config, "TEST_CUTOFF_DATE", None):
        print(f"  Cutoff Date: {config.TEST_CUTOFF_DATE}")
    if getattr(config, "TEST_TIME_WINDOW", None):
        print(f"  Time Window: {config.TEST_TIME_WINDOW}")
    print()
    
def _load_local_metadata(self):
    if not os.path.exists(self.db_config.get("local_metadata_path")):
        return {"last_sync": "", 
                "db_path": "/bigdisk/daniele/cornerstone_module/db"} #DP !!! controlla se non ho creato il file come si comporta
    with open(self.db_config.get("local_metadata_path"), "r") as f:
        return json.load(f)
    
def _save_local_metadata(self, last_sync_date):
    meta_path = self.db_config.get("local_metadata_path")

    # Ensure directory exists
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    # Step 1 — Load existing metadata if present
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            # Corrupted file → start from scratch
            metadata = {}
    else:
        metadata = {}

    # Step 2 — Add missing "db_path" if not present
    if "db_path" not in metadata:
        # Folder path of the metadata file
        metadata["db_path"] = os.path.dirname(meta_path)

    # Step 3 — Update last_sync
    metadata["last_sync"] = last_sync_date

    # Step 4 — Save merged metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
            


def _fetch_remote_metadata(self, output_folder="db"):
    folder_url = self.db_config.get("remote_metadata_url")
    if not folder_url:
        raise RuntimeError("❌ 'remote_metadata_url' not set in db_config.")
    
    os.makedirs(output_folder, exist_ok=True)

    print(f"🔍 Listing files in remote folder: {folder_url}")

    # Download remotely (gdown will create full tree)
    gdown.download_folder(
        url=folder_url,
        output=output_folder,
        quiet=False,
        use_cookies=False
    )

    # Detect JSON files inside output folder
    json_files = [
        f for f in os.listdir(output_folder)
        if f.lower().endswith(".json")
    ]

    if not json_files:
        raise FileNotFoundError("❌ No metadata JSON file found in the downloaded folder.")

    # Pick the first one (or match a prefix if you prefer)
    downloaded_json = os.path.join(output_folder, json_files[0])

    print(f"📄 Found metadata file: {downloaded_json}")

    # Force a consistent filename
    target_json = os.path.join(output_folder, "remote_db_metadata.json")

    # Rename (overwrite if exists)
    os.replace(downloaded_json, target_json)

    print(f"📦 Stored metadata as: {target_json}")

    # Load JSON
    with open(target_json, "r") as f:
        try:
            meta = json.load(f)
            print("✅ Successfully parsed remote metadata.")
            return meta
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON. First 500 chars:")
            print(f.read(500))
            raise
        
        

def _download_file(self, folder_url, dest_folder="db"):
    
    os.makedirs(dest_folder, exist_ok=True)

    # List files before download
    before_files = set(os.path.join(root, f) 
                    for root, _, files in os.walk(dest_folder) 
                    for f in files)

    # Download the folder
    gdown.download_folder(url=folder_url, output=dest_folder, quiet=False, use_cookies=False)

    # List files after download
    after_files = set(os.path.join(root, f) 
                    for root, _, files in os.walk(dest_folder) 
                    for f in files)

    # Determine newly downloaded files
    downloaded_files = after_files - before_files

    if len(downloaded_files) == 0:
        raise FileNotFoundError("❌ No new files were downloaded.")
    elif len(downloaded_files) > 1:
        raise RuntimeError(f"❌ Multiple files were downloaded: {downloaded_files}")
    
    downloaded_file = downloaded_files.pop()
    print(f"✅ Successfully downloaded: {downloaded_file}")


    return  downloaded_file
        

        
def _create_local_db_from_dump(self,  dump_path: str , db_type="sqlite", db_path = None):
    """
    Create or refresh the local MySQL database from a full SQL dump file (.sql or .sql.gz)
    in a memory- and disk-efficient way by streaming the dump directly into MySQL.
    """
    print(f"🛠️  Creating local database from dump: {dump_path}", flush=True)

    
    if db_type == "sqlite":
        """
        Restore an SQLite3 database from a .sql or .sql.gz dump file.
        Uses a temporary file for decompression.
        """
        base_name = os.path.basename(dump_path)
        base_no_ext = base_name.replace(".sql.gz", "").replace(".sql", "")
        db_name = f"{base_no_ext}.db"
        db_path = os.path.join(db_path, db_name)

        # Step 1 — Ensure database file exists
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.close()
            print(f"✅ Database file `{db_path}` is ready.", flush=True)
        except Exception as e:
            print(f"❌ SQLite error creating database file: {e}", flush=True)
            return

        # Step 2 — Prepare temporary SQL file
        print("📦 Preparing SQL dump (temporary file)...", flush=True)
        tmp_mysql_sql = tempfile.NamedTemporaryFile(delete=False, suffix=".sql").name

        try:
            if dump_path.endswith(".gz"):
                with gzip.open(dump_path, "rb") as gz_in, open(tmp_mysql_sql, "wb") as tmp_out:
                    shutil.copyfileobj(gz_in, tmp_out)
            else:
                shutil.copyfile(dump_path, tmp_mysql_sql)

            print(f"✅ Temporary MySQL SQL file: {tmp_mysql_sql}", flush=True)

            # Step 3 — Convert to SQLite SQL (STREAMING, no huge memory load)
            print("🔧 Converting MySQL SQL → SQLite SQL (streaming)...", flush=True)

            tmp_sqlite_sql = tempfile.NamedTemporaryFile(delete=False, suffix=".sql").name

            with open(tmp_mysql_sql, "r", encoding="utf-8", errors="ignore") as fin, \
                open(tmp_sqlite_sql, "w", encoding="utf-8") as fout:

                for line in fin:

                    # Skip MySQL LOCK/UNLOCK TABLES
                    if "LOCK TABLES" in line or "UNLOCK TABLES" in line:
                        continue

                    # Remove /*!40101 ... */
                    if "/*!" in line:
                        continue

                    # Remove ENGINE=..., CHARSET=..., AUTO_INCREMENT
                    for bad in ["ENGINE=", "CHARSET=", "COLLATE=", "AUTO_INCREMENT="]:
                        if bad in line:
                            # Remove everything after first occurrence
                            line = line.split(bad)[0] + ";\n"

                    # Replace backticks with quotes
                    line = line.replace("`", '"')

                    # Fix common MySQL → SQLite type differences
                    replacements = {
                        "tinyint": "INTEGER",
                        "smallint": "INTEGER",
                        "mediumint": "INTEGER",
                        "int(": "INTEGER(",
                        "bigint": "INTEGER",
                        "double": "REAL",
                        "float": "REAL",
                        "decimal": "REAL",
                        "datetime": "TEXT",
                        "timestamp": "TEXT",
                        "text": "TEXT",
                        "longtext": "TEXT",
                        "mediumtext": "TEXT",
                        "blob": "BLOB",
                    }
                    for mysql_t, sqlite_t in replacements.items():
                        safe_pattern = re.escape(mysql_t)
                        line = re.sub(safe_pattern, sqlite_t, line, flags=re.IGNORECASE)

                    # Fix trailing comma before closing parenthesis
                    line = re.sub(r",\s*\)", ")", line)

                    fout.write(line)

            print(f"✅ SQLite-compatible SQL written to: {tmp_sqlite_sql}", flush=True)

            # Step 4 — Import into SQLite
            print("📥 Importing SQL into SQLite...", flush=True)
            subprocess.run(f"sqlite3 {db_path} < {tmp_sqlite_sql}", shell=True, check=True)
            print(f"✅ SQLite database `{db_path}` successfully restored from {dump_path}", flush=True)

        except subprocess.CalledProcessError as e:
            print(f"❌ SQLite restore failed (code {e.returncode})", flush=True)
        except Exception as e:
            print(f"❌ Error: {e}", flush=True)
        finally:
            # Clean up
            for f in [tmp_mysql_sql, locals().get("tmp_sqlite_sql")]:
                if f and os.path.exists(f):
                    os.remove(f)
            print("🧹 Temporary files removed.", flush=True)

        return
        
        
    elif db_type == "Mysql":
            
        db_name = self.db_config.get("database")
        user = self.db_config.get("user")
        password = self.db_config.get("password")
        host = self.db_config.get("host", "localhost")
        port = self.db_config.get("port", 3306)

        # Step 1 — Ensure database exists
        try:
            conn = mysql.connector.connect(host=host, user=user, password=password)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
            print(f"✅ Database `{db_name}` is ready.", flush=True)
        except mysql.connector.Error as err:
            print(f"❌ MySQL error: {err}", flush=True)
            return
        finally:
            try:
                cursor.close()
                conn.close()
            except:
                pass

        # Step 2 — Import dump directly
        print("📥 Importing SQL dump into MySQL (streamed, no temp file)...", flush=True)

        env = os.environ.copy()
        env["MYSQL_PWD"] = password  # safer than passing via -p

        if dump_path.endswith(".gz"):
            # Use gunzip to stream decompressed SQL into mysql
            cmd = f"gunzip -c {dump_path} | mysql -h{host} -P{port} -u{user} {db_name}"
        else:
            # Regular SQL file
            cmd = f"mysql -h{host} -P{port} -u{user} {db_name} < {dump_path}"

        try:
            subprocess.run(cmd, shell=True, check=True, env=env)
            print(f"✅ Database `{db_name}` successfully restored from {dump_path}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ MySQL restore failed (code {e.returncode}):", flush=True)
            if e.stderr:
                print(e.stderr, flush=True)
        except Exception as e:
            print(f"❌ Unexpected error: {e}", flush=True)
        return
            

def _apply_update_to_local_db(self, db_path: str, bin_path: str):
    """
    Apply a database update using a standard SQL script (e.g., generated by sqldiff).
    NOTE: This bypasses the SQLite Session Extension logic due to environment constraints.
    The 'bin_path' argument should point to the generated .sql script.
    """

    # 1. Adapt the input path to the expected SQL file
    # We assume the caller provides the path to the SQL script now.
    sql_path = bin_path
    if sql_path.endswith('.bin'):
        # This is a warning/check, typically you should rename the file externally
        # If the user still passes a .bin path, we assume the corresponding .sql file exists
        sql_path = sql_path.replace('.bin', '.sql')

    print(f"📥 Applying SQL update script: {sql_path} → {db_path}")

    conn = None # Initialize conn outside try block for use in finally/except
    try:
        # Connect to local DB
        conn = sqlite3.connect(db_path)
        # conn.enable_load_extension(True) is no longer needed but harmless

        # --- REMOVED: All conn.load_extension() and changeset_apply() calls ---

        # 2. Read the SQL script file (must be text, not binary)
        with open(sql_path, "r") as f:
            sql_script = f.read()

        # 3. Execute the entire SQL script efficiently
        conn.executescript(sql_script)

        conn.commit()
        # conn.close() will be handled in the finally block

        print("✅ Update applied successfully via SQL script.")

    except FileNotFoundError as e:
        # Catch file errors specifically, which is common if the .sql file is missing
        print(f"❌ Error applying update: SQL update file not found at {sql_path}.")
        raise RuntimeError(
            f"SQL script required at {sql_path}. Ensure sqldiff generated the file."
        ) from e
    
    except Exception as e:
        # Catch all other DB or general errors
        print(f"❌ Error applying update: {e}")
        if conn:
            conn.rollback() # Ensure transaction is rolled back on error
        raise
    
    finally:
        if conn:
            conn.close() 
            
def _get_or_create_task(self, key):
    """
    Utility function: returns a task if already exists, 
    otherwise creates and stores a new ClearML task.
    """
    if key not in self.tasks:
        self.tasks[key] = Task.init(
            project_name=self.project_name,
            task_name=key,
            reuse_last_task_id=False  # Recommended
        )
    return self.tasks[key]


def _generate_plots( summary_metrics, OUTPUT_CONFIG, path_manager, TASK_CONFIG):
    from .ml.utils.plots import (
        plot_boxplot, plot_training_curves,
        plot_metric_curves, plot_test_metrics
    )

    if OUTPUT_CONFIG["plot_boxplot"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            plot_boxplot(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_boxplot.png")
            )

    if OUTPUT_CONFIG["plot_training_curves"]:
        plot_training_curves(
            summary_metrics,
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_loss_curves.png")
        )

    if OUTPUT_CONFIG["plot_metric_curves"]:
        for metric in OUTPUT_CONFIG["plot_metrics"]:
            plot_metric_curves(
                summary_metrics, metric_name=metric,
                save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_{metric}_curve.png")
            )

    if OUTPUT_CONFIG["plot_test_metrics"]:
        plot_test_metrics(
            summary_metrics, selected_metrics=OUTPUT_CONFIG["plot_metrics"],
            save_path=path_manager.get_plot_path(f"{TASK_CONFIG['name']}_test_metrics.png")
        )
        
def _flare_class_from_flux(fflux):
    if fflux >= 1e-4:
        return "X"
    elif fflux >= 1e-5:
        return "M"
    elif fflux >= 1e-6:
        return "C" 
    else:
        return "NO"

'''

def json_to_py_config(json_path, py_path=None):
    """
    Convert a JSON config file to a Python config module.

    Args:
        json_path (str): Path to the input JSON config.
        py_path (str, optional): Path to write the Python module. Defaults to same folder as JSON, named 'config.py'.
    
    Returns:
        str: Path to the generated Python config file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON config not found: {json_path}")
    
    if py_path is None:
        py_path = os.path.join(os.path.dirname(json_path), "config.py")
    
    with open(json_path, "r") as f:
        cfg = json.load(f)
    
    with open(py_path, "w") as f:
        for key, value in cfg.items():
            # Dump each top-level key as a Python variable
            py_value = repr(value)  # converts dict/list/str/int/float into valid Python literal
            f.write(f"{key} = {py_value}\n")
    
    print(f"✅ Generated Python config at: {py_path}")
    return py_path

 
def _fix_google_drive_url(self, url):
    """Convert a Google Drive 'view' URL to a direct download link if needed."""
    if "drive.google.com" in url and "uc?export=download" not in url:
        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return url



def extract_id(self, url: str) -> str:
    """Extract folder or file ID from any Drive URL."""
    m = re.search(r"/(folderview\?id=|drive/folders/|file/d/)([a-zA-Z0-9_-]{25,})", url)
    if not m:
        raise ValueError("Could not find ID in URL")
    return m.group(2)

def extract_folder_id(self, url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]{25,})", url)
    if not m:
        raise ValueError("Invalid folder URL")
    return m.group(1)

def list_folder_metadata(self, folder_id: str, timeout: int = 3) -> List[Tuple[str, str]]:
    """
    Run `gdown --folder` but kill it after `timeout` seconds.
    It prints JSON lines *before* downloading → we capture them.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    cmd = ["gdown", "--folder", url, "--quiet"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError)),
    )
    try:
        # Alarm = hard stop after N seconds
        signal.alarm(timeout)
        stdout, _ = proc.communicate()
        signal.alarm(0)
    except TimeoutError:
        proc.kill()
        stdout, _ = proc.communicate()
        signal.alarm(0)

    files = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "filename" in data and "id" in data:
                    files.append((data["filename"], data["id"]))
            except json.JSONDecodeError:
                continue
    return files
    


     
def convert_mysql_to_sqlite(sql_text: str) -> str:
    """Remove MySQL-specific syntax and rewrite types for SQLite."""
    # Remove MySQL comments like /*!40101 SET ... */
    sql_text = re.sub(r"/\*![0-9]{5}.*?\*/", "", sql_text, flags=re.DOTALL)

    # Remove engine/charset line
    sql_text = re.sub(r"ENGINE=\w+.*?;", ";", sql_text)

    # Remove LOCK/UNLOCK
    sql_text = re.sub(r"LOCK TABLES.*?;", "", sql_text, flags=re.DOTALL)
    sql_text = re.sub(r"UNLOCK TABLES.*?;", "", sql_text, flags=re.DOTALL)

    # Remove AUTO_INCREMENT
    sql_text = re.sub(r"AUTO_INCREMENT=\d+\s*", "", sql_text)

    # Backticks → double quotes
    sql_text = re.sub(r"`([^`]*)`", r'"\1"', sql_text)

    # Type conversions
    type_map = {
        r"\btinyint\b": "INTEGER",
        r"\bsmallint\b": "INTEGER",
        r"\bmediumint\b": "INTEGER",
        r"\bint\b": "INTEGER",
        r"\bbigint\b": "INTEGER",
        r"\bdouble\b": "REAL",
        r"\bfloat\b": "REAL",
        r"\bdecimal\([^)]+\)": "REAL",
        r"\bdatetime\b": "TEXT",
        r"\btimestamp\b": "TEXT",
        r"\btext\b": "TEXT",
        r"\blongtext\b": "TEXT",
        r"\bmediumtext\b": "TEXT",
        r"\bblob\b": "BLOB",
    }
    for mysql_type, sqlite_type in type_map.items():
        sql_text = re.sub(mysql_type, sqlite_type, sql_text, flags=re.IGNORECASE)

    # Remove trailing commas
    sql_text = re.sub(r",\s*\)", ")", sql_text)

    return sql_text
    
'''
