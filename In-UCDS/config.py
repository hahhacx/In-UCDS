neumf_config = {'num_epoch': 100,
                'batch_size': 1024 ,
                'optimizer': 'adam',
                'adam_lr': 1e-4,
                'latent_dim_mf': 32,
                'latent_dim_mlp': 32,
                'num_negative': 4,
                'layers': [64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 1e-5,
                'device_id': 1
                }

pmf_config = {'num_epoch': 100,
              'batch_size': 1024 ,
              'optimizer': 'adam',
              'num_feat': 16,
              'epsilon' : 1,
              '_lambda' : 0.1,
              'momentum': 0.8,
              'maxepoch': 20,
              'num_batches':10,
              'num_negative': 4,
              'l2_regularization': 1e-5,
              'adam_lr': 1e-4
             }

vaecf_config = { 'num_epoch': 100,
                 'batch_size': 1024 ,
                 'optimizer': 'adam',
                 'num_feat': 32,
                 'total_anneal_steps':10000,
                 'anneal_cap':0.2,
                 'dropout':0.5,
                 'num_negative': 4,
                 'l2_regularization': 1e-5,
                 'adam_lr': 1e-4
                }

ngcf_config = { 'num_epoch': 100,
                'batch_size': 1024 ,
                'optimizer': 'adam',
                'embed_size':16,
                'layer_size':[32],
                'adj_type':'norm',
                'alg_type':'ngcf',
                'n_fold':10,
                'node_dropout_flag':True,
                'node_dropout_rate':0.5,
                'decay':1e-5,
                'num_negative': 4,
                'l2_regularization': 1e-5,
                'adam_lr': 1e-4
               }
