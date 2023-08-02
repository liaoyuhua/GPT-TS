import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from .utils import EarlyStopping


class Trainer:
    """
    Default training loss is MAE. If you want to change it, you need to change it in the trainer.
    """

    def __init__(self, model, lr, max_epochs) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.lr = lr
        self.max_epochs = max_epochs

        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr
        )

        self.ealry_stopping = EarlyStopping(patience=10, verbose=True)

    def train(self, train_dataset, val_dataset, batch_size, num_workers):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        for epoch in range(self.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)

            self.ealry_stopping(val_loss, self.model, "checkpoint")

            if self.ealry_stopping.early_stop:
                print("Early stopping")
                break

            print(
                f"Epoch: {epoch+1}/{self.max_epochs} | Train loss: {train_loss} | Val loss: {val_loss} | Time: {time.time()-start_time}"
            )

    def train_epoch(self, train_loader):
        self.model.train()

        total_loss = 0
        for x, y, m in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            m = m.to(self.device)

            self.optimizer.zero_grad()

            _, (lmse, lmae) = self.model(x, m, y)

            lmae.backward()

            self.optimizer.step()

            total_loss += lmae.item()

        return total_loss / len(train_loader)

    def val_epoch(self, val_loader):
        self.model.eval()
        seq_len = val_loader.dataset.seq_len
        pred_len = val_loader.dataset.pred_len

        total_loss = 0
        with torch.no_grad():
            for x, y, m in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                m = m.to(self.device)

                x_enc = x[:, :seq_len, :]
                x_enc_mark = m[:, :seq_len, :]
                x_dec_mark = m[:, seq_len:, :]
                y_true = y[:, seq_len - 1 :, :]

                _, (lmse, lmae) = self.model.predict(
                    x_enc, x_enc_mark, x_dec_mark, y_true
                )

                total_loss += lmae.item()

        return total_loss / len(val_loader)

    def test(self, test_dataset, batch_size, num_workers):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        seq_len = test_dataset.seq_len
        pred_len = test_dataset.pred_len

        self.model.load_state_dict(torch.load("checkpoint/checkpoint.pth"))

        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, y, m in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                m = m.to(self.device)

                x_enc = x[:, :seq_len, :]
                x_enc_mark = m[:, :seq_len, :]
                x_dec_mark = m[:, seq_len:, :]
                y_true = y[:, seq_len:, :]

                _, (lmse, lmae) = self.model.predict(
                    x_enc, x_enc_mark, x_dec_mark, y_true
                )

        return total_loss / len(test_loader)
