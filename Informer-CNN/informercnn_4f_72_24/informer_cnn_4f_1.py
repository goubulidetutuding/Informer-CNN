import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd

from Informer2020_main.utils.masking import TriangularCausalMask, ProbMask
from Informer2020_main.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Informer2020_main.models.decoder import Decoder, DecoderLayer
from Informer2020_main.models.attn import FullAttention, ProbAttention, AttentionLayer
from Informer2020_main.models.embed import DataEmbedding

out_len = 8 # 表示预测长度*2 （比如预测4个点就是8，因为4个经度、4个纬度）
number_pred = 12 # 用于预报的点数
in_len = number_pred
label_len = int(in_len / 2)
# label_len = 0
featuresnum = 14


class OptimizerMaker:
    """Base class that I use to detect if we have an optimizer factory instead
    of an optimizer."""

    def __call__(self, model):
        raise NotImplementedError

class Informer_cnn(nn.Module):
    def __init__(self):
        super(Informer_cnn, self).__init__()
        self.track_model = Informer_Track()
        self.hycom_model = Cnn_Hycom()
        self.fc = nn.Sequential(nn.Linear(out_len+out_len*4, out_len),nn.ReLU())
    def forward(self, x):
        x_track = x[0].squeeze(-1)
        y = x[1].squeeze(-1)
        x_mark = x[2].squeeze(-1)
        y_mark = x[3].squeeze(-1)
        hycom_sst = x[4].squeeze(-1)
        hycom_sal = x[5].squeeze(-1)
        hycom_u = x[6].squeeze(-1)
        hycom_v = x[7].squeeze(-1)

        x_track = x_track.to(torch.float32)
        y = y.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        y_mark = y_mark.to(torch.float32)
        hycom_sst = hycom_sst.to(torch.float32)
        hycom_sal = hycom_sal.to(torch.float32)
        hycom_u = hycom_u.to(torch.float32)
        hycom_v = hycom_v.to(torch.float32)
        x_dec = torch.zeros([y.shape[0], int(out_len/2), y.shape[-1]]).float()
        x_dec = torch.cat([y[:,:label_len,:], x_dec], dim=1).float()

        sst_out = self.hycom_model(hycom_sst)
        sal_out = self.hycom_model(hycom_sal)
        u_out = self.hycom_model(hycom_u)
        v_out = self.hycom_model(hycom_v)
        track_out = self.track_model(x_track,x_mark,x_dec,y_mark)


        mean_out = torch.mean(torch.stack((sst_out, sal_out, u_out, v_out), dim=0), dim=0)
        pred = torch.mean(torch.stack((mean_out,track_out),dim=0),dim=0)


        return pred



class Informer_Track(nn.Module):
    def __init__(self, enc_in=featuresnum, dec_in=featuresnum, c_out=featuresnum, seq_len=in_len, label_len=label_len, out_len=int(out_len/2),
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer_Track, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        x_enc = x_enc.float()
        x_mark_enc = x_mark_enc.float()
        x_dec = x_dec.float()
        x_mark_dec = x_mark_dec.float()


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Cnn_Hycom(nn.Module):
    def __init__(self):
        super(Cnn_Hycom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=number_pred,out_channels=int(out_len/2),kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(out_len/2),out_channels=int(out_len/2),kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=int(out_len/2),out_channels=int(out_len/2),kernel_size=(3,3),stride=(3,2))
        self.fc = nn.Linear(4 * out_len, 4 * featuresnum)

    def forward(self,x):
        x = self.conv1(x) 
        x = self.pool(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = x.view(-1, 4 * out_len)  
        x = self.fc(x)  
        x = x.view(-1, int(out_len/2), featuresnum)  
        return x

#  loss function
class Module(pl.LightningModule):
    def __init__(self, model=None, optimizer=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        # pdb.set_trace()
        y = self.model(x)

        return y

    def training_step(self, batch, batch_idx):

        pred = self.forward(batch)
        loss = self.compute_loss(batch, pred, label='Train')
        # print(loss)
        self.log("Loss_Step/All/Train", loss, on_epoch=False, on_step=True)
        self.log("Loss_Epoch/All/Train", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_id):

        pred = self.forward(batch)
        # pdb.set_trace()
        valid_loss = self.compute_loss(batch, pred, label='validation')
        # print(valid_loss)
        self.log("val_loss", valid_loss, logger=False, on_step=False, on_epoch=True)
        self.log("Loss_Epoch/All/Val", valid_loss, on_epoch=True, on_step=False)

        return {}

    def compute_loss(self, batch, y_pred, label='Train', sync_dist=False, ):

        y_true = batch[1]
        y_true = y_true[:, -int(out_len / 2):, 0:]

        loss = torch.mean((y_pred - y_true) ** 2)
        loss = torch.sqrt(loss)

        self.log(
            f"RPS_Epoch/All/{label}",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=sync_dist, )
        self.log(
            f"RPS_Epoch/All_step/{label}",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=sync_dist, )

        # self.log("Loss_Step/All/olr", loss, on_epoch=False, on_step=True)
        return loss

    def configure_optimizers(self):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=5)
        if isinstance(self.optimizer, OptimizerMaker):
            # return self.optimizer(self.model)
            return {
                'optimizer': self.optimizer(self.model),
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',  # or epoch
                    'frequency': 1,
                },
            }
        else:
            # return self.optimizer
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',  # or epoch
                    'frequency': 1,
                },
            }