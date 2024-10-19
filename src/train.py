"""
Training process
"""
from typing import List, Literal, Dict, Any

import pandas as pd
from rich.progress import track
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils.data_utils import (
    to_dataframe,
    merge_to_ret_df
)
from .model.convnets import Convnet


class TrainSession:
    """
    Training session
    """
    def __init__(
        self,
        _type: Literal["bin", "multi_3", "multi_10"],
        _cls: Literal["SH", "SL", "SM", "BH", "BL", "BM", "ALL"],
        config: Dict[str, Any],
        subsample: List[int]
    ) -> None:
        self.device = \
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.type = _type

        if self.type == "bin":
            self.cls = _cls

        else:
            self.cls = "All"

        self.models = {
            "bin": Convnet(n_classes=2).to(self.device),
            "multi_3": Convnet(n_classes=3).to(self.device),
            "multi_10": Convnet(n_classes=10).to(self.device)   
        }
        self.cfg = config
        self.model = self.models[self.type]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            **self.cfg["optimizer"]
        )
        self._exp_name = f"{self.type}_{self.cls}"
        self.subsample = subsample

    def train(
        self,
        tr_loader: DataLoader,
        vd_loader: DataLoader
    ) -> None:
        """
        train
        """
        stale, best_acc = 0, 0
        train_acc_list, valid_acc_list, train_loss_list, valid_loss_list = [], [], [], []
        for epoch in track(range(self.cfg['n_epochs']), description="Epoch"):
            self.model.train()
            train_loss, train_accs = [], []

            for batch in tr_loader:
                imgs, labels = batch
                labels = labels.long()
                logits = self.model.forward(imgs.to(self.device))
                loss = self.criterion(logits, labels.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = (logits.argmax(dim=-1) == labels.to(self.device)).float().mean()
                train_loss.append(loss.item())
                train_accs.append(acc)

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            train_acc_list.append(train_acc.cpu())
            train_loss_list.append(train_loss)

            print(
                f"[ Train | {epoch + 1:03d} / {self.cfg['n_epochs']:03d} ]|loss = {train_loss:.5f}, acc = {train_acc:.5f}"
            )
            self.model.eval()
            valid_loss, valid_accs = [], []

            for batch in vd_loader:
                imgs, labels = batch
                labels = labels.long()

                with torch.no_grad():
                    logits = self.model.forward(imgs.to(self.device))

                loss = self.criterion(logits, labels.to(self.device))
                acc = (logits.argmax(dim=-1) == labels.to(self.device)).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)

            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            valid_acc_list.append(valid_acc.cpu())
            valid_loss_list.append(valid_loss)

            print(
                f"[ Valid | {epoch + 1:03d} / {self.cfg['n_epochs']:03d} ]|loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
            )

            if valid_acc > best_acc:
                with open(f"model/{self._exp_name}_log.txt", "a", encoding="utf-8"):
                    print(
                        f"[ Valid | {epoch + 1:03d} / {self.cfg['n_epochs']:03d} ]|loss = {valid_loss:.5f}, acc = {valid_acc:.5f} --> best"
                    )
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(self.model.state_dict(), f"model/{self._exp_name}_best.ckpt")
                best_acc = valid_acc
                stale = 0

            else:
                stale += 1
                if stale > self.cfg['patience']:
                    print(f"No improvement {self.cfg['patience']} consecutive epcohs, Early stopping")
                    break


    def merge_label(
        self,
        ts_loader: DataLoader,
        pred_label,
        args
    ) -> None:
        """
        Merge predict probability to return dataframe
        
        data format:
        
        ** OVA_H & OVA_L are temporary unused **
        
        +--------+--------+--------+--------+--------+--------+--------+------------+
        | PERMNO | date   | P_KJX  | P_MC_H | P_MC_L | P_OVA_H| P_OVA_L|  P_FF3_CLS |
        +--------+--------+--------+--------+--------+--------+--------+------------+...
        | 10114  | 200101 |[p1, p2]|  0.097 |  0.096 |  0.101 |  0.097 |[p1, p2, p3]|
        | 10114  | 200102 |[p1, p2]|  0.099 |  0.093 |  0.099 |  0.099 |[p1, p2, p3]|
        ...
        
        Also, the original columns TICKER, DLRET, RET, Volume are remained.
        """
        ret_df = pd.read_parquet("dat/monthly_data.parquet")

        if args.test_env is True:
            imgs = [ts_loader.dataset.imgs[i] for i in self.subsample]

        else:
            imgs = ts_loader.dataset.imgs

        predicted_label_df = to_dataframe(pred_label, imgs)
        ret_df = merge_to_ret_df(predicted_label_df, ret_df)

        if args.labeler == "bin":
            if args.use_spec_mdl:
                ret_df.rename(
                    columns={"label": f"P_KJX_{args.cls}_by{args.spec_mdl}"}, inplace=True
                )
            else:
                ret_df.rename(columns={"label": f"P_KJX_{args.cls}"}, inplace=True)

        elif args.labeler == "tri":
            ret_df.rename(columns={"label": f"P_FF3_{args.cls}"}, inplace=True)

        elif args.labeler.split("-")[0] == "multi":
            if args.labeler.split("-")[1] == "h":
                if args.use_spec_mdl:
                    ret_df.rename(
                        columns={"label": f"P_MC_H_{args.cls}_by{args.spec_mdl}"}, inplace=True
                    )
                else:
                    ret_df.rename(columns={"label": f"P_MC_H_{args.cls}"}, inplace=True)

            elif args.labeler.split("-")[1] == "l":
                if args.use_spec_mdl:
                    ret_df.rename(
                        columns={"label": f"P_MC_L_{args.cls}_by{args.spec_mdl}"}, inplace=True
                    )
                else:
                    ret_df.rename(columns={"label": f"P_MC_L_{args.cls}"}, inplace=True)

        print(ret_df[ret_df[ret_df.columns[-1]].notna()])
        print(ret_df.shape)

        ret_df.to_parquet("dat/monthly_data.parquet", index=False)


    def test(
        self,
        ts_loader: DataLoader,
        sub_ts_loader: DataLoader,
        args
    ) -> None:
        """
        test session
        """
        if args.use_spec_mdl:

            if args.mdl == "bin":
                self.model.load_state_dict(
                    torch.load(f"model/bin_{args.spec_mdl}_best.ckpt")
                )

            elif args.mdl == "multi_10":
                self.model.load_state_dict(
                    torch.load(f"model/multi_10_{args.spec_mdl}_best.ckpt")
                )

        else:
            self.model.load_state_dict(torch.load(f"model/{self._exp_name}_best.ckpt"))

        self.model.eval()
        predicted_label = []

        if args.test_env is True:
            loader = sub_ts_loader
        else:
            loader = ts_loader

        with torch.no_grad():
            for batch in track(loader, description="Testing"):
                imgs, _ = batch
                outputs = self.model(imgs.to(self.device))
                predicted = outputs.cpu().data.numpy()

                if args.labeler == "bin":
                    predicted_label.extend(predicted)

                elif args.labeler == "tri":
                    predicted_label.extend(predicted)

                elif args.labeler == "multi-h":
                    predicted_label.extend(predicted[:, -1])

                elif args.labeler == "multi-l":
                    predicted_label.extend(predicted[:, 0])

        self.merge_label(ts_loader, predicted_label, args)
