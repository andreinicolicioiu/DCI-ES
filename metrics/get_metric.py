from metrics.dci import *
from loss_capacity.train_model_random_forest import train_test_random_forest
import numpy as np

def get_probe_dci(
        probe, dataloader_train, dataloader_val, dataloader_test,
        method='tree_ens_hgb',
        max_leaf_nodes = 200,
        max_depth=20,
        seed = 0, lr = 0.001, epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        device=None,
        data_fraction=1.0):
        # def get_dci(R, scores_train, scores_val, scores_test):

    data_fraction = 0.01 # params.probe.data_fraction
    max_depth= 20 #params.probe.max_depth
    max_leaf_nodes = None
    method='tree_ens_rf'
    pdb.set_trace()
    all_dci = []
    print(f'probe has {len(probe.probe_at_index)} layers')
    print(f'{probe.probe_at_index}')
    for i in range(probe.num_hidden_layers + 1):
        model = probe.probe_at_index[i]
    
        num_probe_params, scores_train, scores_val, scores_test, dci_scores = train_test_random_forest(
                model=model, 
                dataloader_train=dataloader_train, 
                dataloader_val=dataloader_val, 
                dataloader_test=dataloader_test, 
                method=method,
                max_leaf_nodes=None,
                max_depth=max_depth,
                seed = 0,
                lr = 0.1,
                epochs = -123, 
                log_interval =  123,
                save_model = False,
                train_name='probe',
                tb_writer=tb_writer, eval_model=True, savefolder=savefolder,
                device=device,
                data_fraction=data_fraction)
        all_dci.append(dci_scores)
        print(f'results at layer: [{i}]')
        [print(f'{name} : {score}') for name, score in dci_scores.items()]
        
    
    for i, dci in enumerate(all_dci):
        print(f'Results at layer: {i}')
        [print(f'{name} : {score}') for name, score in dci.items()]

    pdb.set_trace()

    #     scores["informativeness_train"] = scores_train
    #     scores["informativeness_val"] = scores_val
    #     scores["informativeness_test"] = scores_test
    #     scores["disentanglement"] = disentanglement(R)
    #     scores["completeness"] = completeness(R)
    return dci_scores#scores


def get_dci(
        model, dataloader_train, dataloader_val, dataloader_test,
        method='tree_ens_hgb',
        max_leaf_nodes = 200,
        max_depth=20,
        seed = 0, lr = 0.001, epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        device=None,
        data_fraction=1.0):
        # def get_dci(R, scores_train, scores_val, scores_test):

    data_fraction = 1000000 #10000 # params.probe.data_fraction
    max_depth= None #params.probe.max_depth
    max_leaf_nodes = None
    method='tree_ens_rf'
    model = model.encode

    num_probe_params, scores_train, scores_val, scores_test, dci_scores = train_test_random_forest(
            model=model, 
            dataloader_train=dataloader_train, 
            dataloader_val=dataloader_val, 
            dataloader_test=dataloader_test, 
            method=method,
            max_leaf_nodes=None,
            max_depth=max_depth,
            seed = 0,
            lr = 0.1,
            epochs = -123, 
            log_interval =  123,
            save_model = False,
            train_name='dci',
            tb_writer=tb_writer, eval_model=True, savefolder=savefolder,
            device=device,
            data_fraction=data_fraction)

    for name, score in dci_scores.items():
        print(f'DCI: {name}: {score} ')
        for i in range(10):
            tb_writer.add_scalar(f'{train_name}-disentanglement/dci_{name}', score, i)

    return dci_scores