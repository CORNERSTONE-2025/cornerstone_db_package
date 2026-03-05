
MODEL_CONFIGS = {  
    "TRANSFORMER": {
    "input_dim": None,
    "model_dim": 64,
    "num_heads": 8,
    "num_layers": 2,
    "dropout_prob": 0.3,
    "temporal_encoding": "none",  # Consistent with ablation studies
    },
    "LSTM": {
    "input_dim": None,
    "model_dim": 128,  # increased for fair comparison (similar capacity to transformer)
    "num_layers": 4,  # increased for fair comparison (similar depth to transformer)
    "bidirectional": True,
    "dropout_prob": 0.3,  # matching transformer dropout
    "pooling": "mask_mean",  # "mask_mean" | "last"
    "use_mask_concatenation": True,  # matching transformer baseline
    "head_hidden": 64,  # add MLP head for fair comparison: 128 -> 64 -> 1
    },
    "MINIROCKET" : {
    "num_kernels": 500  
    }, 
    "ROCKET" : {
    "num_kernels": 5000  
    },
    "XGBOOST" : { 
    "n_estimators" : 100,
    "max_depth" : 3,
    "learning_rate" : 0.1, 
    "mask_concat" : True 
    },
    "INCEPTIONTIME":{
        "input_size": None
    },
    "Autoformer": {############ AUTOFORMER ############
        "model_name": "Autoformer",
        "label_len": 1,
        "task_name": "classification",
        "seq_len": 0,
        "pred_len": 0,
        "moving_avg": 3, #????????????????
        "enc_in":   None,
        "d_model": 128,
        "d_ff": 128 * 2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "dropout": 0.2,
        "factor": 5,
        "n_heads": 4,
        "activation": "relu", #else-->gelu
        "e_layers": 2,
        "dec_in":   None,
        "c_out": 1,
        "d_layers": 2,
        "num_class": 1,
        "use_mask_concatenation": True,
    },
    "DLinear": {############ DLINEAR ############
        "model_name": 'DLinear',
        "label_len": 1,
        "task_name": "classification",
        "seq_len": 16,
        "moving_avg": 3, #????????????????
        "enc_in":   None,
        "num_class": 1,
        "use_mask_concatenation": False,
        "individual": True,
    },
    "ETSformer": {############ ETSFORMER ############
        "model_name": "ETSformer",
        "label_len": 1,
        "task_name": "classification",
        "seq_len": 16,
        "enc_in":   None,
        "d_model": 128,
        "d_ff": 128 * 2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "dropout": 0.2,
        "n_heads": 4,
        "top_k": 2,
        "activation": "relu", #else-->gelu
        "e_layers": 2,
        "c_out": 1,
        "d_layers": 2,
        "num_class": 1,
        "use_mask_concatenation": True,
    },
    "FEDformer": {############ FEDFORMER ############
        "model_name": "FEDformer",
        "label_len": 1,
        "task_name": "classification",
        "seq_len": 16,
        "pred_len": 0,
        "moving_avg": 3, #????????????????
        "enc_in":   None,
        "dec_in":   None,
        "d_model": 128,
        "d_ff": 128 * 2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "dropout": 0.2,
        "n_heads": 4,
        "activation": "relu", #else-->gelu
        "e_layers": 2,
        "c_out": 1,
        "d_layers": 2,
        "num_class": 1,
        "use_mask_concatenation": False,
        "version": 'fourier', #[Fourier, Wavelets]
        "mode_select": 'random', #[random, low]
        "modes": 32,
    },
    "LightTS": {############ LIGHTTS ############
        "model_name": "LightTS",
        "task_name": "classification",
        "enc_in":   None,
        "seq_len": 16,
        "d_model": 128,
        "dropout": 0.2,
        "use_mask_concatenation": False,
        "num_class": 1,
        "chunk_size": 8,
    },
    "Nonstationary_Transformer": {############ NONSTATIONARY_TRANSFORMER ############
        "model_name": "Nonstationary_Transformer",
        "seq_len": 16,
        "pred_len": 0,
        "label_len": 1,
        "task_name": "classification",
        "enc_in":   None,
        "d_model": 128,
        "d_ff": 128 * 2,
        "dropout": 0.2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "e_layers": 2,
        "freq": 'h',
        "n_heads": 4,
        "activation": "relu", #else-->gelu
        "factor": 5,
        "use_mask_concatenation": False,
        "num_class": 1,
        "p_hidden_layers": 3,
        "p_hidden_dims": [128,128,128],  #sono le dimensioni degli hidden layers--|lista?
    },
    "Pyraformer": {############ PYRAFORMER ############
        "model_name": "Pyraformer",
        "seq_len": 16,
        "pred_len": 0,
        "task_name": "classification",
        "enc_in":   None,
        "d_model": 128,
        "dropout": 0.2,
        "use_mask_concatenation": False,
        "num_class": 1,
        "d_ff": 128 * 2,
        "n_heads": 4,
        "e_layers": 2,
        "window_size" :[4,4], 
        "inner_size": 5,
    },
    "Reformer": {############ REFORMER ############
        "model_name": "Reformer",
        "seq_len": 16,
        "pred_len": 0,
        "task_name": "classification",
        "enc_in":   None,
        "d_model": 128,
        "dropout": 0.2,
        "use_mask_concatenation": False,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "n_heads": 4,
        "d_ff": 128 * 2,
        "dropout": 0.2,
        "activation": "relu", #else-->gelu
        "e_layers": 2,
        "bucket_size": 4,
        "n_hashes": 4,
        "num_class": 1,
    },
    "TimesNet": {############ TIMESNET ############
        "model_name": "TimesNet",
        "seq_len": 16,
        "pred_len": 0,
        "top_k": 2,
        "d_model": 128,
        "d_ff": 128 * 2,
        "num_kernels": 3,
        "task_name": "classification",
        "use_mask_concatenation": False,
        "label_len": 1,
        "enc_in":   None,
        "e_layers": 2,
        "dropout": 0.2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "num_class": 1,
    },
    "Informer": {############ INFORMER ############
        "model_name": "Informer",
        "enc_in":   None,
        "dec_in":   None,
        "seq_len": 16,
        "pred_len": 0,
        "d_model": 128,
        "d_ff": 128 * 2,
        "c_out": 1,
        "num_class": 1,
        "dropout": 0.2,
        "embed": 'fixed',  #['fixed', 'learned', 'timeF']
        "freq": 'h',
        "factor": 5,
        "activation": "relu", #else-->gelu
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 2,
        "use_mask_concatenation": False,
        "label_len": 1,
        "distil": None,
        "task_name": "classification",
    },
    "ONEFITSALL": {
        "patch_size" : 8,
        "stride" : 8,
        "d_model" : 768, 
        "dropout" : 0.3, 
        "mask_concat" : True
    },
} 
