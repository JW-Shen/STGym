"""
Script for training and evaluation sota models.
Author: ChunWei Shen
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="MTGNN")
parser.add_argument("--data",type=str,default="metr_la")
args = parser.parse_args()
model = args.model
data = args.data

def main():
    if model == "LSTM":
        if data == "metr_la":
            cmd = "python -m tools.main model=LSTM data=metr_la trainer.lr_skd=null"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=LSTM data=pems_bay trainer.lr_skd=null"
        elif data == "pems03":
            cmd = "python -m tools.main model=LSTM data=pems03 trainer.lr_skd=null"
        elif data == "pems04":
            cmd = "python -m tools.main model=LSTM data=pems04 trainer.lr_skd=null"
        elif data == "pems07":
            cmd = "python -m tools.main model=LSTM data=pems07 trainer.lr_skd=null"
        elif data == "pems08":
            cmd = "python -m tools.main model=LSTM data=pems08 trainer.lr_skd=null"
    elif model == "TCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=TCN data=metr_la trainer.lr_skd=null"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=TCN data=pems_bay trainer.lr_skd=null"
        elif data == "pems03":
            cmd = "python -m tools.main model=TCN data=pems03 trainer.lr_skd=null"
        elif data == "pems04":
            cmd = "python -m tools.main model=TCN data=pems04 trainer.lr_skd=null"
        elif data == "pems07":
            cmd = "python -m tools.main model=TCN data=pems07 trainer.lr_skd=null"
        elif data == "pems08":
            cmd = "python -m tools.main model=TCN data=pems08 trainer.lr_skd=null"
    elif model == "DCRNN":
        if data == "metr_la":
            cmd = "python -m tools.main model=DCRNN data=metr_la trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[20, 30, 40, 50]' trainer.lr_skd.gamma=0.1 trainer.optimizer.lr=0.01\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=DCRNN data=pems_bay trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[20, 30, 40, 50]' trainer.lr_skd.gamma=0.1 trainer.optimizer.lr=0.01\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk"
        elif data == "pems03":
            cmd = "python -m tools.main model=DCRNN data=pems03 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk trainer.dataloader.batch_size=32 data.dp.scaling=minmax"
        elif data == "pems04":
            cmd = "python -m tools.main model=DCRNN data=pems04 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk trainer.dataloader.batch_size=32"
        elif data == "pems07":
            cmd = "python -m tools.main model=DCRNN data=pems07 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk trainer.dataloader.batch_size=32"
        elif data == "pems08":
            cmd = "python -m tools.main model=DCRNN data=pems08 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 +trainer.optimizer.amsgrad=True\
                data.dp.priori_gs.type=dual_random_walk"
    elif model == "STGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=STGCN data=metr_la trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STGCN data=pems_bay trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STGCN data=pems03 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STGCN data=pems04 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STGCN data=pems07 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STGCN data=pems08 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=170"
    elif model == "GWNet":
        if data == "metr_la":
            cmd = "python -m tools.main model=GWNet data=metr_la trainer.lr_skd=null model.model_params.n_series=207\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=GWNet data=pems_bay trainer.lr_skd=null model.model_params.n_series=325\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems03":
            cmd = "python -m tools.main model=GWNet data=pems03 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=358\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems04":
            cmd = "python -m tools.main model=GWNet data=pems04 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=307\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems07":
            cmd = "python -m tools.main model=GWNet data=pems07 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=883\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems08":
            cmd = "python -m tools.main model=GWNet data=pems08 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=170\
                data.dp.priori_gs.type=dbl_transition"
    elif model == "MTGNN":
        if data == "metr_la":
            cmd = "python -m tools.main model=MTGNN data=metr_la trainer.lr_skd=null +trainer.cl.lv_up_period=2500\
                +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=MTGNN data=pems_bay trainer.lr_skd=null +trainer.cl.lv_up_period=2500\
                +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=MTGNN data=pems03 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1500 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=MTGNN data=pems04 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1000 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=MTGNN data=pems07 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1500 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=MTGNN data=pems08 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1000 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=170"
    elif model == "DGCRN":
        if data == "metr_la":
            cmd = "python -m tools.main model=DGCRN data=metr_la trainer/lr_skd=multistep trainer.epochs=150\
                'trainer.lr_skd.milestones=[100, 120]' +trainer.cl.lv_up_period=2500 +trainer.cl.task_lv_max=12\
                data.dp.priori_gs.type=dbl_transition model.model_params.gsl_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=DGCRN data=pems_bay trainer/lr_skd=multistep trainer.epochs=110\
                'trainer.lr_skd.milestones=[100]' +trainer.cl.lv_up_period=6500 +trainer.cl.task_lv_max=12\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition\
                model.model_params.st_params.h_dim=96 model.model_params.st_params.cl_decay_steps=5500\
                model.model_params.gsl_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=DGCRN data=pems03 trainer/lr_skd=multistep trainer.epochs=150\
                'trainer.lr_skd.milestones=[100, 120]' +trainer.cl.lv_up_period=3000 +trainer.cl.task_lv_max=12\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition\
                model.model_params.gsl_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=DGCRN data=pems04 trainer/lr_skd=multistep trainer.epochs=150\
                'trainer.lr_skd.milestones=[100, 120]' +trainer.cl.lv_up_period=3000 +trainer.cl.task_lv_max=12\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition\
                model.model_params.gsl_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=DGCRN data=pems07 trainer/lr_skd=multistep trainer.epochs=150\
                'trainer.lr_skd.milestones=[100, 120]' +trainer.cl.lv_up_period=3000 +trainer.cl.task_lv_max=12\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition\
                model.model_params.gsl_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=DGCRN data=pems08 trainer/lr_skd=multistep trainer.epochs=150\
                'trainer.lr_skd.milestones=[100, 120]' +trainer.cl.lv_up_period=1500 +trainer.cl.task_lv_max=12\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition\
                model.model_params.gsl_params.n_series=170"
    elif model == "STSGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=STSGCN data=metr_la trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STSGCN data=pems_bay trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STSGCN data=pems03 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=16 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STSGCN data=pems04 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STSGCN data=pems07 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=16 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STSGCN data=pems08 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                data.dp.priori_gs.type=binary model.model_params.st_params.n_series=170"
    elif model == "AGCRN":
        if data == "metr_la":
            cmd = "python -m tools.main model=AGCRN data=metr_la trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[10, 20, 30]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.001\
                trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=AGCRN data=pems_bay trainer.lr_skd=null trainer.optimizer.lr=0.003\
                trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=AGCRN data=pems03 trainer.lr_skd=null trainer.es.patience=30\
                trainer.optimizer.lr=0.003 trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=AGCRN data=pems04 trainer.lr_skd=null trainer.es.patience=30\
                trainer.optimizer.lr=0.003 trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=AGCRN data=pems07 trainer.lr_skd=null trainer.es.patience=30\
                trainer.optimizer.lr=0.003 trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=AGCRN data=pems08 trainer.lr_skd=null trainer.es.patience=30\
                trainer.optimizer.lr=0.003 trainer.optimizer.weight_decay=0 model.model_params.st_params.n_series=170"
    elif model == "GMAN":
        if data == "metr_la":
            cmd = "python -m tools.main model=GMAN data=metr_la trainer/lr_skd=multistep trainer.es.patience=10\
                'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]' trainer.lr_skd.gamma=0.7\
                trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/metr_la/SE_metr_la.txt]'"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=GMAN data=pems_bay trainer/lr_skd=multistep trainer.es.patience=10\
                trainer.optimizer.lr=0.0005 'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]'\
                trainer.lr_skd.gamma=0.7 trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True\
                trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems_bay/SE_pems_bay.txt]'"
        elif data == "pems03":
            cmd = "python -m tools.main model=GMAN data=pems03 trainer/lr_skd=multistep trainer.es.patience=10\
                'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]' trainer.lr_skd.gamma=0.7\
                trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems03/SE_pems03.txt]'"
        elif data == "pems04":
            cmd = "python -m tools.main model=GMAN data=pems04 trainer/lr_skd=multistep trainer.es.patience=10\
                'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]' trainer.lr_skd.gamma=0.7\
                trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems04/SE_pems04.txt]'"
        elif data == "pems07":
            cmd = "python -m tools.main model=GMAN data=pems07 trainer/lr_skd=multistep trainer.es.patience=10\
                'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]' trainer.lr_skd.gamma=0.7\
                trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems07/SE_pems07.txt]'"
        elif data == "pems08":
            cmd = "python -m tools.main model=GMAN data=pems08 trainer/lr_skd=multistep trainer.es.patience=10\
                'trainer.lr_skd.milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]' trainer.lr_skd.gamma=0.7\
                trainer.optimizer.weight_decay=0 data.dp.time_enc.add_diw=True trainer.dataloader.batch_size=16\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems08/SE_pems08.txt]'"
    elif model == "GTS":
        if data == "metr_la":
            cmd = "python -m tools.main model=GTS data=metr_la trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=207\
                model.model_params.gsl_params.fc_in_dim=383552\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/metr_la/metr_la.h5]'"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=GTS data=pems_bay trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=325\
                model.model_params.gsl_params.fc_in_dim=583408\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems_bay/pems_bay.h5]'"
        elif data == "pems03":
            cmd = "python -m tools.main model=GTS data=pems03 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=358\
                model.model_params.gsl_params.train_ratio=0.6 model.model_params.gsl_params.fc_in_dim=251312\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems03/pems03.npz]'"
        elif data == "pems04":
            cmd = "python -m tools.main model=GTS data=pems04 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=307\
                model.model_params.gsl_params.train_ratio=0.6 model.model_params.gsl_params.fc_in_dim=162832\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems04/pems04.npz]'"
        elif data == "pems07":
            cmd = "python -m tools.main model=GTS data=pems07 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=883\
                model.model_params.gsl_params.train_ratio=0.6 model.model_params.gsl_params.fc_in_dim=270656\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems07/pems07.npz]'"
        elif data == "pems08":
            cmd = "python -m tools.main model=GTS data=pems08 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.lr_skd.gamma=0.1 'trainer.lr_skd.milestones=[20, 30, 40]' trainer.optimizer.lr=0.005\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 model.model_params.n_series=170\
                model.model_params.gsl_params.train_ratio=0.6 model.model_params.gsl_params.fc_in_dim=171136\
                'data.dp.aux_data_path=[${paths.RAW_DATA_PATH}/pems08/pems08.npz]'"
    elif model == "STNorm":
        if data == "metr_la":
            cmd = "python -m tools.main model=STNorm data=metr_la trainer.lr_skd=null model.model_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STNorm data=pems_bay trainer.lr_skd=null\
                model.model_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STNorm data=pems03 trainer.lr_skd=null model.model_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STNorm data=pems04 trainer.lr_skd=null model.model_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STNorm data=pems07 trainer.lr_skd=null model.model_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STNorm data=pems08 trainer.lr_skd=null model.model_params.n_series=170"
    elif model == "STID":
        if data == "metr_la":
            cmd = "python -m tools.main model=STID data=metr_la trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STID data=pems_bay trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STID data=pems03 trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=358\
                data.dp.scaling=null trainer.rescale=False"
        elif data == "pems04":
            cmd = "python -m tools.main model=STID data=pems04 trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STID data=pems07 trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STID data=pems08 trainer/lr_skd=multistep trainer.epochs=200\
                'trainer.lr_skd.milestones=[1, 50, 80]' trainer.lr_skd.gamma=0.5 trainer.optimizer.lr=0.002\
                trainer.dataloader.batch_size=32 data.dp.time_enc.add_diw=True model.model_params.n_series=170"
    elif model == "SCINet":
        if data == "metr_la":
            cmd = "python -m tools.main model=SCINet data=metr_la trainer/lr_skd=exp trainer.epochs=80\
                trainer.dataloader.batch_size=8 trainer.max_grad_norm=null data.dp.time_enc.add_tid=False"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=SCINet data=pems_bay trainer/lr_skd=exp trainer.epochs=80\
                data.dp.time_enc.add_tid=False trainer.dataloader.batch_size=8 trainer.max_grad_norm=null\
                model.model_params.n_series=325 model.model_params.st_params.dataset_name=pems_bay"
        elif data == "pems03":
            cmd = "python -m tools.main model=SCINet data=pems03 trainer/lr_skd=exp trainer.epochs=80\
                data.dp.time_enc.add_tid=False trainer.dataloader.batch_size=8 trainer.max_grad_norm=null\
                model.model_params.n_series=358 model.model_params.st_params.dataset_name=pems03"
        elif data == "pems04":
            cmd = "python -m tools.main model=SCINet data=pems04 trainer/lr_skd=exp trainer.epochs=80\
                data.dp.time_enc.add_tid=False trainer.dataloader.batch_size=8 trainer.max_grad_norm=null\
                model.model_params.n_series=307 model.model_params.st_params.n_decoder_layer=1\
                model.model_params.st_params.dropout=0 model.model_params.st_params.dataset_name=pems04"
        elif data == "pems07":
            cmd = "python -m tools.main model=SCINet data=pems07 trainer/lr_skd=exp trainer.epochs=80\
                data.dp.time_enc.add_tid=False trainer.dataloader.batch_size=8 trainer.max_grad_norm=null\
                model.model_params.n_series=883 model.model_params.st_params.n_decoder_layer=1\
                model.model_params.st_params.h_ratio=0.03125 model.model_params.st_params.dataset_name=pems07"
        elif data == "pems08":
            cmd = "python -m tools.main model=SCINet data=pems08 trainer/lr_skd=exp trainer.epochs=80\
                data.dp.time_enc.add_tid=False trainer.dataloader.batch_size=8 trainer.max_grad_norm=null\
                model.model_params.n_series=170 model.model_params.st_params.n_decoder_layer=1\
                model.model_params.st_params.dropout=0.5 model.model_params.st_params.h_ratio=1\
                model.model_params.st_params.dataset_name=pems08"
    elif model == "STAEformer":
        if data == "metr_la":
            cmd = "python -m tools.main model=STAEformer data=metr_la trainer/lr_skd=multistep trainer.epochs=200\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[20, 30]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=30 trainer.optimizer.weight_decay=0.0003 trainer.dataloader.batch_size=16\
                data.dp.time_enc.add_diw=True model.model_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STAEformer data=pems_bay trainer/lr_skd=multistep trainer.epochs=300\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[10, 30]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.dataloader.batch_size=16 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STAEformer data=pems03 trainer/lr_skd=multistep trainer.epochs=300\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[15, 30, 40]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.weight_decay=0.0005 trainer.dataloader.batch_size=16\
                data.dp.time_enc.add_diw=True trainer.loss_fn._target_=torch.nn.HuberLoss ~trainer.loss_fn.name\
                model.model_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STAEformer data=pems04 trainer/lr_skd=multistep trainer.epochs=300\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[15, 30, 50]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.weight_decay=0.0005 trainer.dataloader.batch_size=16\
                data.dp.time_enc.add_diw=True trainer.loss_fn._target_=torch.nn.HuberLoss ~trainer.loss_fn.name\
                model.model_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STAEformer data=pems07 trainer/lr_skd=multistep trainer.epochs=300\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[15, 35, 50]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.0005 trainer.optimizer.weight_decay=0.001\
                trainer.dataloader.batch_size=4 data.dp.time_enc.add_diw=True ~trainer.loss_fn.name\
                trainer.loss_fn._target_=torch.nn.HuberLoss model.model_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STAEformer data=pems08 trainer/lr_skd=multistep trainer.epochs=300\
                trainer.max_grad_norm=null 'trainer.lr_skd.milestones=[25, 45, 65]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=30 trainer.optimizer.weight_decay=0.0015 trainer.dataloader.batch_size=16\
                data.dp.time_enc.add_diw=True trainer.loss_fn._target_=torch.nn.HuberLoss ~trainer.loss_fn.name\
                model.model_params.n_series=170"
    elif model == "MegaCRN":
        if data == "metr_la":
            cmd = "python -m tools.main model=MegaCRN data=metr_la trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.01 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=MegaCRN data=pems_bay trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.01 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=MegaCRN data=pems03 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.01 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=MegaCRN data=pems04 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.01 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=MegaCRN data=pems07 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.005 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 trainer.dataloader.batch_size=32 model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=MegaCRN data=pems08 trainer/lr_skd=multistep trainer.epochs=200\
                trainer.custom_loss=True 'trainer.lr_skd.milestones=[50, 100]' trainer.lr_skd.gamma=0.1\
                trainer.es.patience=20 trainer.optimizer.lr=0.01 trainer.optimizer.weight_decay=0\
                trainer.optimizer.eps=1e-3 model.model_params.st_params.n_series=170"
    elif model == "PGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=PGCN data=metr_la trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=PGCN data=pems_bay trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
        elif data == "pems03":
            cmd = "python -m tools.main model=PGCN data=pems03 trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
        elif data == "pems04":
            cmd = "python -m tools.main model=PGCN data=pems04 trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
        elif data == "pems07":
            cmd = "python -m tools.main model=PGCN data=pems07 trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
        elif data == "pems08":
            cmd = "python -m tools.main model=PGCN data=pems08 trainer.lr_skd=null\
                trainer.dataloader.batch_size=32 data.dp.priori_gs.type=dbl_transition"
    elif model == "STIDGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=STIDGCN data=metr_la trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=500 trainer.es.patience=30 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=207 model.model_params.st_params.h_dim=64"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STIDGCN data=pems_bay trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=500 trainer.es.patience=30 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=325 model.model_params.st_params.h_dim=64"
        elif data == "pems03":
            cmd = "python -m tools.main model=STIDGCN data=pems03 trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=300 trainer.es.patience=100 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STIDGCN data=pems04 trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=500 trainer.es.patience=100 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=307 model.model_params.st_params.h_dim=64"
        elif data == "pems07":
            cmd = "python -m tools.main model=STIDGCN data=pems07 trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=500 trainer.es.patience=30 trainer.dataloader.batch_size=16\
                data.dp.time_enc.add_diw=True model.model_params.n_series=883 model.model_params.st_params.h_dim=128"
        elif data == "pems08":
            cmd = "python -m tools.main model=STIDGCN data=pems08 trainer/optimizer=ranger trainer.lr_skd=null\
                trainer.epochs=500 trainer.es.patience=100 data.dp.time_enc.add_diw=True\
                model.model_params.n_series=170 model.model_params.st_params.h_dim=96"

    os.system(cmd)

if __name__ == "__main__":
    main()