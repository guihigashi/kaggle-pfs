import os
import pickle
import tempfile

import click
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from kaggle_pfs.data import readers


class SalesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, begin_month=0, end_month=32, target_month=33):
        self.shop = torch.tensor(df["shop_id"].values)
        self.item = torch.tensor(df["item_id"].values)
        # columns from 0 (first month) to 32 (second to last month)
        self.sales = torch.tensor(
            df.loc[:, map(str, range(begin_month, end_month + 1))].values,
            dtype=torch.float32,
        )
        # month 33 (last month)
        self.target = torch.tensor(df[str(target_month)].values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (
            self.shop[index],
            self.item[index],
            self.sales[index, :],
        ), self.target[index]


class SalesDataloader:
    def __init__(self, device, dataloader):
        self.device = device
        self.dl = dataloader

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)

        for xb, yb in batches:
            # just move to device
            shop = xb[0].to(self.device)
            item = xb[1].to(self.device)

            # reshape for LSTM layer
            sales = xb[2].unsqueeze(2).transpose(0, 1).to(self.device)

            # make it an array
            target = yb.view(-1, 1).to(self.device)

            yield (shop, item, sales), target


class Network(nn.Module):
    def __init__(
        self,
        shop_embedding_dim=2,
        item_embedding_dim=2,
        hidden_dim=12,
        dense_dim=24,
    ):
        super().__init__()
        self.shop_embedding = nn.Embedding(60, shop_embedding_dim, max_norm=2)
        self.item_embedding = nn.Embedding(22170, item_embedding_dim, max_norm=2)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim)
        self.linear_1 = nn.Linear(
            shop_embedding_dim + item_embedding_dim + hidden_dim,
            dense_dim,
        )
        self.linear_2 = nn.Linear(dense_dim, 1)

    def forward(self, shop, item, sales):
        shop = self.shop_embedding(shop)
        item = self.item_embedding(item)
        sales, _ = self.lstm(sales)
        x = torch.cat((shop, item, sales[-1]), dim=1)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x


def loss_function(*args, **kwargs):
    return torch.sqrt(F.mse_loss(*args, **kwargs))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(*xb), yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


def write_2d_embedding_weights(emb, name, epoch):
    weights = emb.weight.detach().cpu().numpy()
    df = pd.DataFrame(weights, columns=["x", "y"])
    df["epoch"] = epoch
    df.to_csv(
        readers.data_path("interim", f"emb-{name}-{epoch}.csv"),
        index_label=name,
    )


@click.command()
@click.option(
    "--epochs",
    default=15,
    help="Number of epochs to fit the model",
)
@click.option(
    "--train_split",
    default=0.75,
    help="Fraction of data in the trainning set",
)
@click.option(
    "--batch_size",
    default=128,
    help="Number of row in each batch size",
)
def main(epochs, train_split, batch_size):
    load_dotenv()

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "epochs": epochs,
                "train_split": train_split,
                "batch_size": batch_size,
            }
        )

        output_dir = tempfile.mkdtemp()
        writer = SummaryWriter(log_dir=output_dir)
        print(f"Writing TensorBoard events locally to: {output_dir}")

        def write_scalars(metrics, epoch):
            writer.add_scalars(
                main_tag=run.info.run_id, tag_scalar_dict=metrics, global_step=epoch
            )
            mlflow.log_metrics(metrics=metrics, step=epoch)

        # training config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get dataset
        dataset = SalesDataset(readers.items_by_month())

        # split dateset
        n_train = int(len(dataset) * train_split)
        n_test = len(dataset) - n_train
        train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

        # dataloader
        train_dataloader = SalesDataloader(
            device, DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
        )
        val_dataloader = SalesDataloader(
            device, DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        )

        # get network
        model = Network(
            shop_embedding_dim=2,
            item_embedding_dim=2,
            hidden_dim=6,
            dense_dim=6,
        ).to(device)

        # get optimizer and link with network params
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # train loop
        for epoch in range(epochs):
            model.train()
            for i, (xb, yb) in enumerate(train_dataloader):
                loss = loss_batch(model, loss_function, xb, yb, optimizer)
                if i % 100 == 0:
                    mlflow.log_metric("batch_loss", value=loss)

            model.eval()
            with torch.no_grad():
                train_loss = np.mean(
                    [
                        loss_batch(model, loss_function, xb, yb)
                        for xb, yb in train_dataloader
                    ]
                )
                valid_loss = np.mean(
                    [
                        loss_batch(model, loss_function, xb, yb)
                        for xb, yb in val_dataloader
                    ]
                )
                write_scalars(
                    {
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                    },
                    epoch,
                )

            write_2d_embedding_weights(model.shop_embedding, "shop", epoch)
            write_2d_embedding_weights(model.item_embedding, "item", epoch)

        # add_graph() will trace the sample input through your model,
        # and render it as a graph.
        writer.add_graph(model, xb)
        writer.flush()

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\n"
            f"tensorboard --logdir={os.path.join(mlflow.get_artifact_uri(), 'events')}"
        )

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch-model",
            pickle_module=pickle,
        )
        print(
            f"\nThe model is logged at: {os.path.join(mlflow.get_artifact_uri(), 'pytorch-model')}"
        )


if __name__ == "__main__":
    main()
