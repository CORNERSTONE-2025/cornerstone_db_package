import json
from cornerstone import Cornerstone

with open("cornerstone/configs/db_config.json") as f:
    db_cfg = json.load(f)

cs = Cornerstone(db_cfg)
''' 
cs.db_update()  
''' 
'''
for tp in ["DEF","NRT"]:
    for cut in ["Cplus", "Mplus","Xplus"]:
        cs.splitting_labeling( shtyp=tp,class0=cut,save_csv=True, data_version="clean")
''' 

for tp in ["NRT"]: #"DEF",
    for cut in ["Cplus", "Mplus","Xplus"]:
        cs.create_hdf5_from_db(
                label_type = cut,
                data_version="all",
                table_type=tp,
                window_hours=24,
                buffer_minutes=6,
                sampling_interval=12,
                test_mode=False,
                test_size=50
            )    
'''
'''

'''
split_config = json.load(open("split_config.json"))


cs.create_dataset_splits( split_config ) 
'''


'''
cs.train_models( train_config_path="train_config.json")
'''
#"./dataset_compilation/data/sharp_dataset_db_SHARPC_clean_DEF_w24h_s12min.h5",


#cs.test_model(test_config_path="train_config.json")

#cs.training(parametri..) #config.py per i parametri 
#cs.predict()
#cs.plot()

