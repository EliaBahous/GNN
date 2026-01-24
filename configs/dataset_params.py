
# CITESEER DATASET OPT
CITECEER_OPT = {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 
       'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 32, 
       'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'exp_kernel', 'augment': False, 
       'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 
       'data_norm': 'rw', 'dataset': 'Citeseer', 'decay': 0.1, 'directional_penalty': None, 'dropout': 0.7488085003122172, 
       'dt': 0.001, 'dt_min': 1e-05, 'epoch': 20, 'exact': True, 'fc_out': False, 'function': 'laplacian',
       'gdc_avg_degree': 64, 'gdc_k': 128, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0,
       'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.6803233752085334, 'jacobian_norm2': None, 
       'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.5825086997804176, 'lr': 0.00863585231323069, 'max_epochs': 1000,
       'max_iters': 100, 'max_nfe': 3000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'Citeseer_beltrami_1_KNN',
       'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_class': 6, 'num_init': 2,
       'num_nodes': 2120, 'num_samples': 400, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 
       'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 4, 'regularise': False,
       'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False,
       'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True,
       'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 7.874113442879092, 'tol_scale': 2.9010446330432815, 'tol_scale_adjoint': 1.0, 
       'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': True, 'use_lcc': True, 'use_mlp': False, 'no_early': True}


# PUBMED DATASET OP
PUBMED_OPT = {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': True, 'adjoint_method': 'adaptive_heun', 
       'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1.0, 'attention_dim': 128, 'attention_norm_idx': 0,
         'attention_rewiring': False, 'attention_type': 'cosine_sim', 'augment': False, 'baseline': False, 'batch_norm': False, 
         'beltrami': False, 'beta_dim': 'sc', 'block': 'rewire_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Pubmed', 
         'decay': 0.0005, 'directional_penalty': None, 'dropout': 0.3, 'dt': 0.001, 
         'dt_min': 1e-05, 'epoch': 600, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 
         'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk',
           'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 128,
             'input_dropout': 0.3, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 
             'lr': 0.005, 'max_epochs': 150, 'max_iters': 100, 'max_nfe': 100000, 'method': 'dopri5',
               'metric': 'test_acc', 'mix_features': False, 'name': None, 'new_edges': 'random', 'no_alpha_sigmoid': False,
                 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 8, 'ode_blocks': 1, 'optimizer': 'adam', 
                 'patience': 50, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 10,
                   'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 
                   'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.005, 
                   'rw_rmvR': 0.005, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1,
                     'threshold_type': 'addD_rvR', 'time': 12.942327880200853, 'tol_scale': 1e-3, 
                     'tol_scale_adjoint': 1e-3, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 
                     'use_labels': True, 'use_lcc': True, 'use_mlp': False, 'folder': 'pubmed_linear_att_beltrami_adj2', 'index': 0,
                       'run_with_KNN': False, 'change_att_sim_type': False, 'reps': 1, 'max_test_steps': 100, 'no_early': True,
                         'earlystopxT': 5.0, 'pos_enc_csv': False, 'pos_enc_type': 'GDC'}


#CORA DATASET OPT
CORA_OPT = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10, 'not_lcc': True, 'dataset': 'Cora', 'force_reload': True,
        'attention_norm_idx': 0, 'simple': True, 'alpha': 0, 'alpha_dim': 'sc', 'beta_dim': 'sc', "use_labels": True,
        'hidden_dim': 64, 'block': 'attention', 'function': 'laplacian', 'alpha_sigmoid': True, 'augment': False, 'adjoint': False,
        'tol_scale': 70, 'time': 20, 'input_dropout': 0.5, 'dropout': 0.2, 'method': 'dopri5', 'optimizer':'adam', 'lr':0.009, "use_mlp": True,
        'decay':0.007, 'epoch':20, 'kinetic_energy':None, 'jacobian_norm2':None, 'total_deriv':None, 'directional_penalty':None, "beltrami": False,
        'no_early': True, 'fc_out': False, 'batch_norm': False, 'heads': 8, 'attention_dim': 128, 'attention_type': 'scaled_dot', 'label_rate': 0.5,
        'square_plus': True, 'reweight_attention': False, 'step_size':1, 'max_nfe':5000, 'no_alpha_sigmoid':False, 'add_source':False }


opt = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10, 'not_lcc': True, 'dataset': 'Cora', 'force_reload': True,
        'attention_norm_idx': 0, 'simple': True, 'alpha': 0, 'alpha_dim': 'sc', 'beta_dim': 'sc', "use_labels": True,
        'hidden_dim': 64, 'block': 'rewire_attention', 'function': 'laplacian', 'alpha_sigmoid': True, 'augment': False, 'adjoint': False,
        'tol_scale': 70, 'time': 20, 'input_dropout': 0.5, 'dropout': 0.2, 'method': 'dopri5', 'optimizer':'adam', 'lr':0.008, "use_mlp": True,
        'decay':0.007, 'epoch':20, 'kinetic_energy':None, 'jacobian_norm2':None, 'total_deriv':None, 'directional_penalty':None, "beltrami": False}
opt["batch_norm"] = False
opt["heads"] = 8
opt["attention_dim"] = 128
opt['attention_type'] = 'scaled_dot'
opt['label_rate'] = 0.5
opt['square_plus'] = True
opt['reweight_attention'] = False
opt['step_size'] = 1
opt['max_nfe'] = 5000
opt['no_alpha_sigmoid'] = False
opt['add_source'] = False
opt['fc_out'] = False
opt['att_samp_pct'] = 1