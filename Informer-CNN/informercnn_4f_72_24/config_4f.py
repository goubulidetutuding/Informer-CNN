import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import importlib


class Config():
    def __init__(self):
        # 制作训练数据集参数
        self.trainfile_path = "/home/ly/xia_datan/wangying/trace/_72_24/fore72_track_72_24_14f.csv"

        # 制作训练数据集对应的时间id表
        self.time_id_path = "/home/ly/xia_datan/wangying/trace/_72_24/fore72_track_72_24_time.csv"

        self.path = "./model_outputs_informer_trace14f/"

        # hycom 制作数据集  需替换
        self.sst_path = "/home/ly/xia_datan/wangying/trace/_72_24/hycom_track_sst_72_24.csv"
        self.sal_path = "/home/ly/xia_datan/wangying/trace/_72_24/hycom_track_sal_72_24.csv"
        self.u_path = "/home/ly/xia_datan/wangying/trace/_72_24/hycom_track_u_72_24.csv"
        self.v_path = "/home/ly/xia_datan/wangying/trace/_72_24/hycom_track_v_72_24.csv"
        # self.sst_path = "/home/ly/xia_datan/bouydataTrace/data/hycom_track_sst_12_24.csv"

        # hycom数据格式转换  25和13表示抽取hycom数据时抽取lon和lat抽取出来的点数   25*13=325
        self.hycom_x = 25
        self.hycom_y = 13

        # 用于预报的点数
        self.numbers_pred = 12
        # 特征数
        self.featuresnum = 14
        self.in_len = self.numbers_pred
        self.out_len = 8  # self.out_len表示预测长度*2 （比如预测4个点就是8，因为4个经度、4个纬度）

        self.label_len = int(self.in_len / 2)  # informer中的label_len
        # self.label_len = 0

        self.nn1 = 0.98
        self.nn2 = 0.90

        self.target_optimizer = getattr(importlib.import_module("Funcs_train_hycom_4f"), "adam")

        self.lr_hycom = 5e-5

        # test
        self.batch_size_infer = 1
        self.batch_size_hycom = 128

        self.target_model_hycom= getattr(importlib.import_module("informer_cnn_4f_1"), "Informer_cnn")
        self.target_module_hycom = getattr(importlib.import_module("informer_cnn_4f_1"), "Module")

        self.filepick_hycom = f"{self.path}/"

        self.experiment_hycom = {
            "user": os.environ.get("USER"),
            "logging": {
                "experiment_name": "track_xiarx",
                "run_name": "Informer_Cnn",
                "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", f"file:{self.filepick_hycom}/ml-runs"),
                "slurm_job_id": os.getenv("SLURM_JOB_ID", None),
                "slurm_array_jobid": os.getenv("SLURM_ARRAY_JOB_ID", None),
                "slurm_array_taskid": os.getenv("SLURM_ARRAY_TASK_ID", None),
            },
            "dataset": {
                "include_model": False,
            },
            "limit_train_batches": 1.0,
            "lr_find": False,
            "early_stopping":{"patience": 20},
        }


        self.cfg_ba_hycom = {
            'num_workers': int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
            'accelerator': 'auto',
            'gpus': 0,
            'accumulate_grad_batches': 1,
            'env': None,
            "max_epoch": 100,
            "gradient_clip_val":0.0,
        }

        self.infer_checkpoint_path = f"{self.path}/tensorboard"
        self.output_file = "/home/ly/xia_datan/bouydataTrace/infer/test.nc"

def find_trainmodle(checkpoint_path):
    max_timedir = 0
    max_timedir_name = ""

    for root, dirs, files in os.walk(checkpoint_path):
        for name in dirs:
            a = os.path.join(root, name)
            a_time = os.path.getctime(a)
            if a_time > max_timedir:
                max_timedir = a_time
                max_timedir_name = a
    checkpoint_path = f"{max_timedir_name}/{os.listdir(max_timedir_name)[0]}"
    return checkpoint_path









