"""
CNN model and training session
"""
import os
import warnings
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
from .utils import filter_imgs, filter_files
from .trans_label import to_dataframe, merge_to_ret_df
warnings.filterwarnings("ignore")
plt.style.use("ggplot")


class StockDataset(Dataset):
    """
    transform the data into tensor
    
    attributes:
    - path: path to the dataset
    - label_path: path to the label
    - tfm: torchvision.transforms
    - add_noise: add noise to the image
    
    Special methods:
    - torchvision.transforms
        Do no augmentation instead only transform to torch.Tensor

    - add_noise:
        Generate two random indices from (0, 59), and replace the black pixel with white pixel
    """
    def __init__(
            self, img_path: str, label_path: str, tfm, add_noise: bool, _cls: str, imgs=None
        ) -> None:
        self.cls = _cls
        self.label_data = filter_imgs(img_path, label_path, self.cls)
        self.imgs = filter_files(img_path, label_path, self.cls)
        if imgs is not None:
            self.imgs = imgs
        self.tfm = tfm
        self.add_noise = add_noise


    def __len__(self) -> int:
        return len(self.imgs)


    def __getitem__(self, idx: int) -> tuple:
        im = Image.open(self.imgs[idx])
        im = self.tfm(im)
        if self.add_noise:
            random_idx = np.random.choice(len(im[0]), 2)
            while im[0][random_idx[0]][random_idx[1]] == 0:
                im[0][random_idx[0]][random_idx[1]] = 1
        label = self.label_data[
            self.label_data["fig_name"] == self.imgs[idx].split(".")[0].split("/")[-1]
        ]["new_label"].values[0]
        return im, label


class ConvNetForBin(nn.Module):
    """
    Convolutional Neural Network for binary classification
    
    Input shape: [1, 60, 60]
    
    Matrix shape (Conv layer):
    - Conv2d(1, 64, (5, 3), 1) --> [64, 56, 58]
    - MaxPool2d(kernel_size=(2, 1)) --> [64, 28, 58]
    - Conv2d(64, 128, (5, 3), 1) --> [128, 24, 56]
    - MaxPool2d(kernel_size=(2, 1)) --> [128, 12, 56]
    - Conv2d(128, 256, (5, 3), 1) --> [256, 8, 54]
    - MaxPool2d(kernel_size=(2, 1)) --> [256, 4, 54]
    
    Matrix shape (Fully connected layer):
    - Linear(256 * 4 * 54, 1024) --> [1024]
    - Linear(1024, 512) --> [512]
    - Linear(512, 128) --> [128]
    - Linear(128, 64) --> [64]
    - Linear(64, 2) --> [2]
    
    Softmax() --> to probability
    """
    def __init__(self) -> None:
        super(ConvNetForBin, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (5, 3), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, (5, 3), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 256, (5, 3), 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.dropout = nn.Sequential(nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 54, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 2),
            nn.Softmax()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward prop
        """
        x = self.cnn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNetForMulti(ConvNetForBin, nn.Module):
    """
    Convolutional Neural Network for multi-classification with 10 classes
    """
    def __init__(self, classes: int=10) -> None:
        super(ConvNetForMulti, self).__init__()
        self.classes = classes
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 54, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 64),
            nn.Linear(64, self.classes),
            nn.Softmax()
        )


class TrainSession:
    """
    Training session
    """
    def __init__(
            self, tr_loader, vd_loader, ts_loader, sub_ts_loader,
            _type: str, _cls: str, subsample: list
        ) -> None:
        self.device = \
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.type = _type
        if self.type == "bin":
            self.cls = _cls
        else:
            self.cls = "All"
        self.models = {
            "bin": ConvNetForBin().to(self.device),
            "multi_3": ConvNetForMulti(classes=3).to(self.device),
            "multi_10": ConvNetForMulti(classes=10).to(self.device)   
        }
        self.model = self.models[self.type]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-5,
            momentum=0.9,
            weight_decay=1e-5,
        )
        self.tr_loader = tr_loader
        self.vd_loader = vd_loader
        self.sub_ts_loader = sub_ts_loader
        self.ts_loader = ts_loader
        self.cfg = {
            "n_epochs": 10,
            "_exp_name": f"{self.type}_{self.cls}",
            "patience": 5
        }
        self.subsample = subsample


    def plot_eval(
            self, train_loss_list: list, valid_loss_list: list,
            train_acc_list: list, valid_acc_list: list
        ) -> None:
        """
        plot the evaluation
        """
        _, axs = plt.subplots(2, 1, figsize=(16, 9))
        axs[0].plot(train_loss_list, label="train loss", color="red")
        axs[0].plot(valid_loss_list, label="valid loss", color="blue")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(train_acc_list, label="train acc", color="red")
        axs[1].plot(valid_acc_list, label="valid acc", color="blue")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        plt.show()
        plt.savefig(f"{os.getcwd()}{self.cfg['n_epochs']}_epochs.png")


    def train(self) -> None:
        """
        train
        """
        stale, best_acc = 0, 0
        train_acc_list, valid_acc_list, train_loss_list, valid_loss_list = [], [], [], []
        for epoch in range(self.cfg['n_epochs']):
            self.model.train()
            train_loss, train_accs = [], []

            for batch in tqdm(self.tr_loader):
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

            for batch in tqdm(self.vd_loader):
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
                with open(f"model/{self.cfg['_exp_name']}_log.txt", "a", encoding="utf-8"):
                    print(
                        f"[ Valid | {epoch + 1:03d} / {self.cfg['n_epochs']:03d} ]|loss = {valid_loss:.5f}, acc = {valid_acc:.5f} --> best"
                    )
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(self.model.state_dict(), f"model/{self.cfg['_exp_name']}_best.ckpt")
                best_acc = valid_acc
                stale = 0

            else:
                stale += 1
                if stale > self.cfg['patience']:
                    print(f"No improvement {self.cfg['patience']} consecutive epcohs, Early stopping")
                    break


    def merge_label(self, pred_label, args) -> None:
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
            imgs = [self.ts_loader.dataset.imgs[i] for i in self.subsample]
        else:
            imgs = self.ts_loader.dataset.imgs
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
        elif args.labeler == "multi-h":
            ret_df.rename(columns={"label": "P_MC_H"}, inplace=True)
        elif args.labeler == "multi-l":
            ret_df.rename(columns={"label": "P_MC_L"}, inplace=True)
        print(ret_df[ret_df[f"P_KJX_{args.cls}"].notna()])
        print(ret_df.shape)
        # elif args.labeler == "ova-h":
        #     ret_df.rename(columns={"label": "P_OVA_H"}, inplace=True)
        # elif args.labeler == "ova-l":
        #     ret_df.rename(columns={"label": "P_OVA_L"}, inplace=True)
        ret_df.to_parquet("dat/monthly_data.parquet", index=False)


    def test(self, args) -> None:
        """
        test session
        """
        if args.use_spec_mdl:
            self.model.load_state_dict(torch.load(f"model/bin_{args.spec_mdl}_best.ckpt"))
        else:
            self.model.load_state_dict(torch.load(f"model/{self.cfg['_exp_name']}_best.ckpt"))
        self.model.eval()
        predicted_label = []
        loader = self.ts_loader
        if args.test_env is True:
            loader = self.sub_ts_loader
        with torch.no_grad():
            for batch in tqdm(loader):
                imgs, _ = batch
                outputs = self.model(imgs.to(self.device))
                # predicted = np.argmax(outputs.cpu().data.numpy(), axis=1) ## label
                if args.labeler == "bin":
                    predicted = outputs.cpu().data.numpy() ## probability
                    predicted_label.extend(predicted)
                elif args.labeler == "tri":
                    predicted = outputs.cpu().data.numpy() ## probability
                    predicted_label.extend(predicted)
                elif args.labeler == "multi-h":
                    h_predicted = outputs.cpu().data.numpy()[:, -1]
                    predicted_label.extend(h_predicted)
                elif args.labeler == "multi-l":
                    l_predicted = outputs.cpu().data.numpy()[:, 0]
                    predicted_label.extend(l_predicted)
                # elif args.labeler == "ova-h":
                #     predicted = outputs.cpu().data.numpy()[:, -1]
                #     predicted_label.extend(predicted)
                # elif args.labeler == "ova-l":
                #     predicted = outputs.cpu().data.numpy()[:, -1]
                #     predicted_label.extend(predicted)

        self.merge_label(predicted_label, args)


def load_config(config_path: str) -> dict:
    """
    load config
    """
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def parse_args() -> argparse.ArgumentParser:
    """
    parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add_noise", type=bool, default=False,
        help="add noise to the image"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="train or test"
    )
    parser.add_argument(
        "--labeler", type=str, default="bin", help="bin, tri, multi-h, multi-l"
    )
    parser.add_argument(
        "--mdl", type=str, default="bin", help="bin, multi_3, multi_10"
    )
    parser.add_argument(
        "--cls", type=str, default="SH",
        help="The class of Fama French 3 factors, SH, SL, SM, BL, BH, BM, ALL"
    )
    parser.add_argument(
        "--use_spec_mdl", type=bool, default=False,
        help="Use specific model for confusing FF3 class"
    )
    parser.add_argument(
        "--spec_mdl", type=str, default="SH",
        help="Specific pre-trained model for confusing FF3 class"
    )
    # parser.add_argument(
    #     "--config", type=str, help="config file"
    # )
    parser.add_argument(
        "--test_env", type=bool, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # model_config = load_config(_args.config)
    ## TODO: Has not setup config file yet
    train_dataset_dir = f"{os.getcwd()}/dat/train/"
    test_dataset_dir = f"{os.getcwd()}/dat/test/"
    train_label_dir = f"{os.getcwd()}/label/train_label_{_args.mdl}_v2.csv"
    test_label_dir = f"{os.getcwd()}/label/test_label_{_args.mdl}_v2.csv"
    train_dataset = StockDataset(
        train_dataset_dir, train_label_dir, transform, _args.add_noise, _args.cls
    )
    test_dataset = StockDataset(
        test_dataset_dir, test_label_dir, transform, _args.add_noise, _args.cls
    )
    train_size, valid_size = \
        int(0.7 * len(train_dataset)), len(train_dataset) - int(0.7 * len(train_dataset))
    train_set, valid_set = random_split(train_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    subsample = list(range(0, 1000, 1))
    sub_test = torch.utils.data.Subset(test_dataset, subsample)
    sub_test_loader = DataLoader(sub_test, batch_size=128, shuffle=False)
    train_session = TrainSession(
        train_loader, valid_loader, test_loader, sub_test_loader,
        _args.mdl, _args.cls, subsample=subsample
    )
    if _args.mode == "train":
        print("Start Training...")
        train_session.train()
    elif _args.mode == "test":
        print("Start Testing...")
        train_session.test(_args)
