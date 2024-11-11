import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import logging
import xarray as xr
import pandas as pd
import tqdm
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as loggers
import torch
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from config_4f import Config
import matplotlib.pyplot as plt
import config_4f
from datetime import datetime
from draw_plot_4f import draw_baseline,draw_box,draw_error
from Funcs_train_hycom_4f import makeDataset,makeDataset_trace,makehycom_dataset

_logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy("file_system")

class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        super().__init__(**kwargs)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint):
        if (self.best_model_score is not None) and (self.logger is not None):
            self.logger.experiment.log_metric(f"{self.monitor}_min", self.best_model_score) 
        return super().on_save_checkpoint(trainer, pl_module, checkpoint) 


def run_experiment(trainpara,
        train_X, train_y,train_X_mark,train_y_mark,
        valid_X, valid_y,valid_X_mark,valid_y_mark,
        train_hycom_sst,valid_hycom_sst,
        train_hycom_sal,valid_hycom_sal,
        train_hycom_u,valid_hycom_u,
        train_hycom_v,valid_hycom_v,
        batch_size,
        num_workers,
        lr_find):


    train_dataset = makeDataset(train_X, train_y, train_X_mark, train_y_mark,
                                train_hycom_sst,train_hycom_sal,train_hycom_u,train_hycom_v)
    valid_dataset = makeDataset(valid_X, valid_y, valid_X_mark, valid_y_mark,
                                valid_hycom_sst,valid_hycom_sal,valid_hycom_u,valid_hycom_v)

    print(f"Length of datasets. Train: {len(train_dataset)}. Val: {len(valid_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=num_workers,
                                               batch_sampler=None,
                                               persistent_workers=True,
                                               )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               batch_sampler=None,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               persistent_workers=True, )

    model = trainpara.target_model_hycom()
    adam = trainpara.target_optimizer
    optimizer = adam(model, lr=trainpara.lr_hycom)

    module = trainpara.target_module_hycom
    lightning_module = module(model=model,optimizer=optimizer)
    tensorboard = loggers.TensorBoardLogger(f"{trainpara.filepick_hycom}/tensorboard", default_hp_metric=False)

    mlflow = loggers.MLFlowLogger(
        trainpara.experiment_hycom["logging"]["mlflow_uri"], 
        run_name=trainpara.experiment_hycom["logging"]["run_name"], 
        tracking_uri=trainpara.experiment_hycom["logging"]["mlflow_uri"], 
        tags={
            "user": trainpara.experiment_hycom["user"], 
            "slurm_job_id": trainpara.experiment_hycom["logging"]["slurm_job_id"], 
            "slurm_array_task_id": trainpara.experiment_hycom["logging"]["slurm_array_taskid"], 
            "slurm_array_job_id": trainpara.experiment_hycom["logging"]["slurm_array_jobid"], 
            "cwd": os.getcwd(), 
        }
    )

    mlflow.log_hyperparams(trainpara.experiment_hycom)

    if "checkpoint_monitor" in trainpara.experiment_hycom:
        checkpoint_monitor = trainpara.experiment_hycom["checkpoint_monitor"]
    else:
        checkpoint_monitor = "val_loss"

    checkpointer = ModelCheckpoint(monitor=checkpoint_monitor)

    other_callbacks = []
    if "callbacks" in trainpara.experiment_hycom:
        for callback_dict in trainpara.experiment_hycom["callbacks"]:
            other_callbacks.append(callback_dict)

    callbacks = [checkpointer,
                 pl_callbacks.LearningRateMonitor(),
                 *other_callbacks,]

    if "early_stopping" in trainpara.experiment_hycom:
        early_stopping = pl_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=trainpara.experiment_hycom["early_stopping"]["patience"]
        )
        callbacks.append(early_stopping)

    # training
    trainer = pl.Trainer(
        log_every_n_steps=1, 
        logger=[tensorboard,mlflow], 
        callbacks=callbacks, 
        default_root_dir=f"{trainpara.filepick_hycom}/logging/lightning/", 
        accelerator=trainpara.cfg_ba_hycom["accelerator"], 
        max_epochs=trainpara.cfg_ba_hycom["max_epoch"], 
        accumulate_grad_batches=trainpara.cfg_ba_hycom["accumulate_grad_batches"], 
        limit_train_batches=trainpara.experiment_hycom["limit_train_batches"], 
        enable_progress_bar=True, 
    )

    if lr_find:
        lr_finder = trainer.tuner.lr_find(
            lightning_module, train_dataloader, valid_dataloader
        ) 
        lr_finder.results 
        fig = lr_finder.plot(suggest=True) 
        filename = "lr.png"
        plt.savefig(filename)
        _logger.info(f"Saved LR curve: {os.getcwd() + '/' + filename}.")
    else:
        print("11111")
        valid_dataloader = valid_dataloader if len(valid_dataloader) > 0 else None 
        trainer.fit(lightning_module,train_dataloader,valid_dataloader)
        best_score = float(checkpointer.best_model_score.cpu())
        mlflow.log_metrics({"val_loss_min": best_score})


def textdata_readmodel(test_dataset,checkpoint_path,trainpara):
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  batch_sampler=None,
                                                  shuffle=False,
                                                  num_workers=int(trainpara.cfg_ba_hycom["num_workers"]))

    model = trainpara.target_model_hycom()
    adam = trainpara.target_optimizer
    optimizer = adam(model, lr=trainpara.lr_hycom)

    module = trainpara.target_module_hycom
    lightning_module = module(model=model, optimizer=optimizer)
    checkpoint = torch.load(checkpoint_path)

    lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)
    lightning_module.eval()
    lightning_module.freeze()

    dataset_of_examples = []
    for example in tqdm.tqdm(test_dataloader):
        pytorch_example = lightning_module(example)
        pytorch_example = pytorch_example.numpy()
        dataset_of_examples.append(pytorch_example)
    _logger.info(f"Outputting forecast to {trainpara.output_file} . ")
    aaa = np.array(dataset_of_examples)
    print(aaa.shape)

    return dataset_of_examples

def write_nc(dataset_of_example, all_data, checkpoint_path, trainpara):


    dataset_of_example = np.array(dataset_of_example).squeeze(1)
    pred_y = dataset_of_example.reshape(dataset_of_example.shape[0], -1)
    pred_y_1 = y_scaler1.inverse_transform(pred_y)  

    test_X_re = test_X.reshape(test_X.shape[0], -1)
    test_X_re = x_scaler1.inverse_transform(test_X_re)

    pred_y_lat = pred_y_1[:, ::trainpara.featuresnum]
    pred_y_lon = pred_y_1[:, 1::trainpara.featuresnum]

    test_x_lat = test_X_re[:, (trainpara.in_len - 1) * trainpara.featuresnum]
    test_x_lon = test_X_re[:, (trainpara.in_len - 1) * trainpara.featuresnum + 1]

    pred_result_lat = pred_y_lat
    pred_result_lon = pred_y_lon

    test_y_1 = test_y[:, -4:, :]
    test_y_1 = test_y_1.reshape(test_y_1.shape[0], -1)
    test_y_1 = y_scaler1.inverse_transform(test_y_1)
    lat_test_y = test_y_1[:, ::trainpara.featuresnum]  # lat
    lon_test_y = test_y_1[:, 1::trainpara.featuresnum]  # lon
    result_test_lat = lat_test_y
    result_test_lon = lon_test_y

    data_11 = pd.read_csv(trainpara.trainfile_path, header=None).values
    nn1_11 = trainpara.nn1
    n1_11 = int(len(data_11) * nn1_11)
    datapick_11 = data_11[n1_11:, :]
    endslice_11 = trainpara.numbers_pred * trainpara.featuresnum
    true_lat_x_11 = datapick_11[:, 0:endslice_11][:, ::trainpara.featuresnum]
    true_lon_x_11 = datapick_11[:, 0:endslice_11][:, 1::trainpara.featuresnum]
    true_lat_y_11 = datapick_11[:, endslice_11:endslice_11 + trainpara.out_len][:, ::2]
    true_lon_y_11 = datapick_11[:, endslice_11:endslice_11 + trainpara.out_len][:, 1::2]

    ds = xr.Dataset(
        {
            "predictions_lon": (["cases", "time"], pred_result_lon),
            "predictions_lat": (["cases", "time"], pred_result_lat),

            "test_lon": (["casex", "time"], result_test_lon),
            "test_lat": (["cases", "time"], result_test_lat),

            "true_lat_nodiff": (["cases", "time"], true_lat_y_11),
            "true_lon_nodiff": (["cases", "time"], true_lon_y_11),

            "true_lon_nodiff_x": (["cases", "time1"], true_lon_x_11),
            "true_lat_nodiff_x": (["cases", "time1"], true_lat_x_11),
        },
        # coords={"cases": range(final_pred.shape[0]), "feature": range(final_pred.shape[1])},
    )

    folder_path = os.path.dirname(checkpoint_path)
    filename = f"{folder_path}/tracecon_{datetime.now().strftime('%Y%m%d%H%M%S')}.nc"
    # filename = f"/home/ly/xia_datan/wangying/trace/trace/test/test_{datetime.now().strftime('%Y%m%d%H%M%S')}.nc"
    ds.to_netcdf(filename)

    return filename



if __name__ == '__main__':

    trainpara = Config()
    train_X,train_y,valid_X,valid_y,test_X,test_y,x_scaler1,y_scaler1,train_X_mark, train_y_mark,valid_X_mark, valid_y_mark,test_X_mark, test_y_mark,all_data = makeDataset_trace(
        trainfile_path=trainpara.trainfile_path,
        time_id_path=trainpara.time_id_path,
        n_features=trainpara.featuresnum,
        in_len=trainpara.in_len,
        out_len=trainpara.out_len,
        nn1=trainpara.nn1,
        nn2=trainpara.nn2,
        label_len = trainpara.label_len)

    print(f"test_y.shape:{test_y.shape}")
    print(f"test_y_mark.shape:{test_y_mark.shape}")

    train_hycom_sst, valid_hycom_sst, test_hycom_sst = makehycom_dataset(trainpara.sst_path,trainpara.numbers_pred,
                                                                         trainpara.nn1,trainpara.nn2,
                                                                         trainpara.hycom_x,trainpara.hycom_y,)
    train_hycom_sal, valid_hycom_sal, test_hycom_sal = makehycom_dataset(trainpara.sal_path, trainpara.numbers_pred,
                                                                         trainpara.nn1, trainpara.nn2,
                                                                         trainpara.hycom_x, trainpara.hycom_y)

    train_hycom_u, valid_hycom_u, test_hycom_u = makehycom_dataset(trainpara.u_path, trainpara.numbers_pred,
                                                                   trainpara.nn1, trainpara.nn2, trainpara.hycom_x,
                                                                   trainpara.hycom_y)

    train_hycom_v, valid_hycom_v, test_hycom_v = makehycom_dataset(trainpara.v_path, trainpara.numbers_pred,
                                                                   trainpara.nn1, trainpara.nn2, trainpara.hycom_x,
                                                                   trainpara.hycom_y)
    print(f"test_hycom_sst.shape:{test_hycom_sst.shape}")


    print(f"Working directory: {os.getcwd()}")
    _logger.info(f"Working directory: {os.getcwd()}")

    # run_experiment(
    #     trainpara,
    #     train_X, train_y,train_X_mark,train_y_mark,
    #     valid_X, valid_y,valid_X_mark,valid_y_mark,
    #     train_hycom_sst,valid_hycom_sst,
    #     train_hycom_sal,valid_hycom_sal,
    #     train_hycom_u,valid_hycom_u,
    #     train_hycom_v,valid_hycom_v,
    #     batch_size=trainpara.batch_size_hycom,
    #     num_workers=int(trainpara.cfg_ba_hycom["num_workers"]),
    #     lr_find=trainpara.experiment_hycom["lr_find"],
    # )

    # 找到本次训练的模型存放路径
    # checkpoint_path = config_4f.find_trainmodle(trainpara.infer_checkpoint_path)
    # print(checkpoint_path)
    # 指定模型存放路径时需把上面一行代码注释掉，并根据实际需求填写checkpoint_path
    checkpoint_path = "/home/ly/model_outputs_informer_trace14f_uv/tensorboard/lightning_logs/version_0/checkpoints/epoch=35-step=24372.ckpt"

    test_dataset = makeDataset(test_X, test_y, test_X_mark, test_y_mark,test_hycom_sst,test_hycom_sal,test_hycom_u,test_hycom_v)
    dataset_of_example = textdata_readmodel(test_dataset, checkpoint_path, trainpara)

    # 将测试集结果存入nc文件中
    ncname = write_nc(dataset_of_example, all_data, checkpoint_path, trainpara)
    print(ncname)
    #
    # # # ncname = "/home/ly/xia_datan/wangying/trace/track_models_hycom/model_outputs/tensorboard/lightning_logs/version_0/checkpoints/tracecon_20240718220021.nc"
    # print("正在绘制baseline...")
    # data_all = draw_baseline(ncname, len(test_X), trainpara)
    # print("baseline绘制完成")
    #
    # # 绘制boxplot
    # draw_box(data_all[0], data_all[3], data_all[6], "Lon", ncname)
    # draw_box(data_all[1], data_all[4], data_all[7], "Lat", ncname)
    # draw_box(data_all[2], data_all[5], data_all[8], "Distance", ncname)
    # print("box绘制完成")
    #
    # # 绘制误差图
    # draw_error(data_all, ncname)
    # print("误差绘制完成")






















