import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler

from ts_benchmark.baselines.ialv_ad_transformer.models.IALV_AD_Transformer_model import (
    ALV_AD_Transformer as IALV_ADTransformerModel,
)
from ts_benchmark.baselines.alv_ad_transformer.ALV_AD_Transformer import (
    ALV_AD_Transformer as BaseALV_ADTransformer,
)
from ts_benchmark.baselines.alv_ad_transformer.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split


class ALV_AD_Transformer(BaseALV_ADTransformer):
    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        del test_data

        self.detect_hyper_param_tune(train_data)
        self.model = IALV_ADTransformerModel(self.config)
        self.model.to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        self.train_eval_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = None
        if getattr(self.config, "lradj", "type1") == "TST":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=len(self.train_data_loader),
                pct_start=getattr(self.config, "pct_start", 0.3),
                epochs=self.config.num_epochs,
                max_lr=self.config.lr,
            )

        for epoch in range(self.config.num_epochs):
            self.model.train()

            for input, _ in self.train_data_loader:
                self.optimizer.zero_grad()
                input = input.float().to(self.device)
                output = self.model(input, None, None, None)
                loss = self.criterion(output, input)
                loss.backward()
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config, scheduler)


class IALV_AD_Transformer(ALV_AD_Transformer):
    pass
