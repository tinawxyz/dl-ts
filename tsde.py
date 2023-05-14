import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from util import *

class TSDataset(Dataset):
    def __init__(self, data, ycol, task, ftrn, nprd, nseq):
        self.nseq = nseq
        self.nprd = nprd
        ntrn = int(len(data) * ftrn)
        xcol = [c for c in data.columns if c != ycol]
        
        if task == 'train':
            self.y = data[ycol].iloc[:ntrn].values
            self.x = data[xcol].iloc[:ntrn, :].values
        else:
            self.y = data[ycol].iloc[ntrn:].values
            self.x = data[xcol].iloc[ntrn:, :].values
        
        self.scaler = StandardScaler()
        self.scaler.fit(data[xcol].iloc[:ntrn, :])
        self.x = self.scaler.transform(self.x)
        
    def __getitem__(self, idx):
        xbeg = idx
        xend = idx + self.nseq + self.nprd
        yprvbeg = idx
        yprvend = idx + self.nseq
        ybeg = idx + self.nseq + 1
        yend = idx + self.nseq + self.nprd + 1
        return self.x[xbeg:xend], self.y[yprvbeg:yprvend], self.y[ybeg:yend]
    
    def __len__(self):
        return len(self.x) - self.nseq - self.nprd

class Res(nn.Module):
    def __init__(self, idim, hdim, odim, drop):
        super(Res, self).__init__()
        self.lin1 = nn.Linear(idim, hdim)
        self.lin2 = nn.Linear(hdim, odim)
        self.lin3 = nn.Linear(idim, odim)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(odim)
        
    def forward(self, x): # x: (nbat, ?, idim)
        h = self.drop(self.lin2(F.relu(self.lin1(x))))
        r = self.lin3(x)
        return self.norm(h + r)

class Enc(nn.Module):
    def __init__(self, idim, covariatedim, hdim, drop, nlay, nprd, nseq):
        super(Enc, self).__init__()
        self.featproj = Res(idim, covariatedim, covariatedim, drop)
        self.enc1 = Res((nseq + nprd) * covariatedim + nseq, hdim, hdim, drop)
        self.enc2 = nn.ModuleList([Res(hdim, hdim, hdim, drop) for _ in range(nlay - 1)])
        self.nlay = nlay
        
    def forward(self, x, yprv): # x: (nbat, nseq + nprd, idim), yprv: (nbat, nseq)
        self.covariate = self.featproj(x) # covariate: (nbat, nseq + nprd, covariatedim)
        self.covariate_flatten = torch.flatten(self.covariate, start_dim=1) # covariate_flatten: (nbat, (nseq + nprd) * covariatedim)
        h = torch.cat([self.covariate_flatten, yprv], dim = 1) # h: (nbat, (nseq + nprd) * covariatedim + nseq)
        h = self.enc1(h)
        for i in range(self.nlay - 1):
            h = self.enc2[i](h) # h: (nbat, hdim)
        return h
    
class Dec(nn.Module):
    def __init__(self, idim, covariatedim, hdim, odim, drop, nlay, nprd):
        super(Dec, self).__init__()
        self.dec1 = nn.ModuleList([Res(hdim, hdim, hdim, drop) for _ in range(nlay - 1)])
        self.dec2 = Res(hdim, hdim, nprd * odim, drop)
        self.featproj = Res(idim, covariatedim, covariatedim, drop)
        self.temporal = Res(odim + covariatedim, hdim, 1, drop)
        self.odim = odim
        self.nlay = nlay
        self.nprd = nprd
        
    def forward(self, h, x):
        for i in range(self.nlay - 1):
            h = self.dec1[i](h) # h: (nbat, hdim)
        self.g = self.dec2(h) # g: (nbat, nprd * odim)
        self.g_unflatten = self.g.view((self.g.size(0), self.nprd, self.odim))
        self.covariate_prd = self.featproj(x[:, -self.nprd:, :])
        o = torch.cat([self.g_unflatten, self.covariate_prd], dim = -1) # o: (nbat, nprd, odim + covariatedim)
        o = self.temporal(o) # o: (nbat, nprd, 1)
        return o

class TSDE(nn.Module):
    def __init__(self, idim, covariatedim, hdim, odim, drop, nlay, nprd, nseq):
        super(TSDE, self).__init__()
        self.enc = Enc(idim, covariatedim, hdim, drop, nlay, nprd, nseq)
        self.dec = Dec(idim, covariatedim, hdim, odim, drop, nlay, nprd)
        self.res = nn.Linear(nseq, nprd)
        
    def forward(self, x, yprv): # x: (nbat, nseq + nprd, idim), yprv: (nbat, nseq)
        h = self.enc(x, yprv)
        o = self.dec(h, x)
        p = o.squeeze(-1) + self.res(yprv) # p: (nbat, nprd)
        return p

class EarlyStop:
    def __init__(self, patience, savepath):
        self.patience = patience
        self.savepath = savepath
        self.minloss = None
        self.count = 0
        self.stop = False
        
    def __call__(self, loss, model):
        if self.minloss is None:
            self.save(model)
            self.minloss = loss
        elif loss >= self.minloss:
            self.count += 1
            print('patience %d minloss:' % self.count, self.minloss)
            if self.count >= self.patience:
                self.stop = True
        else:
            self.minloss = loss
            self.count = 0
            self.save(model)
            
    def save(self, model):
        torch.save(model.state_dict(), '%s/best.state' % self.savepath)
        torch.save(model, '%s/best.model' % self.savepath)

class Trial:
    def __init__(self, idim, covariatedim, hdim, odim, drop, nlay, nprd, nseq, patience, savepath):
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.device = torch.device('cuda:0')
            print('use gpu')
        else:
            self.device = torch.device('cpu')
            print('use cpu')
        self.model = TSDE(idim, covariatedim, hdim, odim, drop, nlay, nprd, nseq).float().to(device)
        self.earlystop = EarlyStop(patience, savepath)
        os.makedirs(savepath, exist_ok=True)
        self.savepath = savepath
        self.lossfn = nn.MSELoss()

        param = {'idim': idim, 'covariatedim': covariatedim, 'hdim': hdim, 'odim': odim,
                 'drop': drop, 'nlay': nlay, 'nprd': nprd, 'nseq': nseq}
        json.dump(param, open('%s/model.param' % savepath, 'w'))

    def train(self, dlt, dlv, lr, nepoch):
        with open('%s/scale.txt' % self.savepath, 'w') as fo:
            fo.write(' '.join(['%f' % f for f in dlt.dataset.scaler.scale_]))
        with open('%s/mean.txt' % self.savepath, 'w') as fo:
            fo.write(' '.join(['%f' % f for f in dlt.dataset.scaler.mean_]))

        losst, lossv = [], []
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        for e in range(nepoch):
            self.model.train()
            losses = []
            for i, (x, yprv, y) in enumerate(dlt):
                optim.zero_grad()
                x = x.float().to(self.device)
                yprv = yprv.float().to(self.device)
                y = y.float().to(self.device)
                p = self.model(x, yprv)
                loss = self.lossfn(p, y)
                losses.append(loss.item())
                loss.backward()
                optim.step()       
                if i % 100 == 0:
                    print('epoch %d batch %d train loss:' % (e, i), losses[-1])
            print('epoch %d train loss:' % e, np.mean(losses))
            losst.extend(losses)
            lossv.append(self.valid(dlv))
            self.earlystop(lossv[-1], self.model)
            if self.earlystop.stop:
                print('earlystop')
                break

        self.model.load_state_dict(torch.load('%s/best.state' % self.earlystop.savepath))
        torch.save(optim.state_dict(), '%s/optim.state' % self.savepath)

        for startidx in [0, 50, 100, 150]:
            plt.plot(np.arange(len(losst[startidx:])), losst[startidx:])
            plt.title('train loss[%d:]' % startidx)
            plt.savefig('trainloss%d.png' % startidx)

        plt.plot(np.arange(len(lossv)), lossv)
        plt.title('valid loss')
        plt.savefig('validloss.png')
        return self.model, losst, lossv

    def valid(self, dlv):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, yprv, y) in enumerate(dlv):
                x = x.float().to(self.device)
                yprv = yprv.float().to(self.device)
                y = y.float().to(self.device)
                p = self.model(x, yprv)
                losses.append(self.lossfn(p, y))
        lossv = np.mean(losses)
        print('valid loss:', lossv)
        self.model.train()
        return lossv

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--years', type=str, default='2020,2021,2022')
    parser.add_argument('--cols', type=str, default='open,close,lowopen,highopen')
    parser.add_argument('--minrow', type=int, default=1110000)
    parser.add_argument('--ccy', type=str, default='GBPAUD')
    parser.add_argument('--ftrn', type=float, default=0.8)
    parser.add_argument('--nprd', type=int, default=3)
    parser.add_argument('--nseq', type=int, default=10)
    parser.add_argument('--nbat', type=int, default=512)
    parser.add_argument('--covariatedim', type=int, default=15)
    parser.add_argument('--hdim', type=int, default=10)
    parser.add_argument('--odim', type=int, default=5)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--nlay', type=int, default=2)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--savepath', type=str, default='./checkpoint')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nepoch', type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = '%s_%s_%s_np%d_ns%d_nb%d_cd%d_hd%d_od%d_nl%d_lr%s_ne%d' % \
        (args.ccy, args.years.split(',')[0], args.years.split(',')[-1], args.nprd, args.nseq,
         args.nbat, args.covariatedim, args.hdim, args.odim, args.nlay, args.lr, args.nepoch)
    savepath = '%s/%s' % (args.savepath, name)

    df = getCcy(args.years.split(','), args.cols.split(','), args.minrow)
    dst = TSDataset(data=df, ycol=args.ccy, task='train', ftrn=args.ftrn, nprd=args.nprd, nseq=args.nseq)
    dsv = TSDataset(data=df, ycol=args.ccy, task='valid', ftrn=args.ftrn, nprd=args.nprd, nseq=args.nseq)
    print(len(dst), len(dsv))
    dlt = DataLoader(dst, batch_size=args.nbat, shuffle=True, drop_last=True)
    dlv = DataLoader(dsv, batch_size=args.nbat, shuffle=True, drop_last=True)

    idim = dst.x.shape[1]
    print(idim)
    trial = Trial(idim=idim, covariatedim=args.covariatedim, hdim=args.hdim, odim=args.odim, drop=args.drop,
          nlay=args.nlay, nprd=args.nprd, nseq=args.nseq, patience=args.patience, savepath=savepath)
    trial.train(dlt, dlv, args.lr, args.nepoch)

