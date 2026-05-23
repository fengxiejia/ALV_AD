import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.alv_ad_transformer.models.ALV_AD_Transformer_model import (
    ALV_AD_Transformer as ALV_ADTransformerModel,
)
from ts_benchmark.baselines.alv_ad_transformer.utils.score import (
    derive_quantizer_time_scores,
    zscore,
)
from ts_benchmark.baselines.alv_ad_transformer.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.alv_ad_transformer.utils.window import (
    window_scores_to_sequence,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split


DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 10,
    "patience": 3,
    "seq_len": 100,
    "horizon": 0,
    "lradj": "type1",
    "pct_start": 0.3,
    "d_model": 256,
    "d_ff": 1024,
    "e_layers": 2,
    "n_heads": 8,
    "n_streams": 4,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "score_modes": "recon_stage1",
    "hybrid_score_lambda": 1.0,
    "rvq_num_embeddings": 128,
    "rvq_grad_scale": 0.25,
    "rvq_gate_floor": 0.2,
    "rvq_gate_temperature": 4.0,
    "frequency_loss_weight": 0.0,
    "frequency_loss_type": "mag",
}


DEFAULT_SCORE_MODES = ("recon_stage1",)


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class ALV_AD_Transformer:
    def __init__(self, **kwargs):
        super(ALV_AD_Transformer, self).__init__()
        score_modes = kwargs.pop("score_modes", DEFAULT_SCORE_MODES)
        self.config = TransformerConfig(**kwargs)
        if isinstance(score_modes, str):
            score_modes = [item.strip() for item in score_modes.split(",") if item.strip()]
        self.score_modes = tuple(score_modes) if score_modes else DEFAULT_SCORE_MODES
        self.config.score_modes = ",".join(self.score_modes)

        self.model_name = self.__class__.__name__
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.seq_len = self.config.seq_len

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception:
            freq = "S"
        if freq is None:
            raise ValueError("Irregular time intervals")
        if freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48
        self.config.task_name = "anomaly_detection"

    def detect_validate(self, valid_data_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.float().to(self.device)
                output = self.model(input, None, None, None)
                loss = criterion(output, input).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def _training_loss_components(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_loss = self.criterion(output, target)
        frequency_loss = output.new_tensor(0.0)
        loss = time_loss
        frequency_weight = float(getattr(self.config, "frequency_loss_weight", 0.0))

        if frequency_weight:
            frequency_type = getattr(self.config, "frequency_loss_type", "mag")
            if frequency_type != "mag":
                raise ValueError("ALV_AD frequency_loss_type only supports 'mag'.")
            output_freq = torch.fft.fft(output, dim=1)
            target_freq = torch.fft.fft(target, dim=1)
            frequency_loss = (output_freq.abs() - target_freq.abs()).abs().mean()
            loss = loss + frequency_weight * frequency_loss
        return loss, time_loss, frequency_loss

    def _training_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _, _ = self._training_loss_components(output, target)
        return loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        del test_data

        self.detect_hyper_param_tune(train_data)
        self.model = ALV_ADTransformerModel(self.config)
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

        train_steps = len(self.train_data_loader)
        iter_count = 0
        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            self.model.train()
            non_finite_loss = False

            for i, (input, _) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                input = input.float().to(self.device)
                output = self.model(input, None, None, None)
                loss, time_loss, frequency_loss = self._training_loss_components(output, input)
                if not torch.isfinite(loss):
                    print(
                        f"Non-finite training loss at epoch {epoch + 1}; "
                        "stop current run and restore the best finite checkpoint."
                    )
                    non_finite_loss = True
                    break
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f}".format(
                            i + 1,
                            epoch + 1,
                            time_loss.item(),
                            frequency_loss.item(),
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if non_finite_loss:
                break

            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config, scheduler)

        if self.early_stopping.check_point is None:
            raise RuntimeError("ALV_AD_Transformer training did not produce a finite checkpoint.")

    def _collect_score_components(self, data_loader):
        self.model.eval()
        criterion = nn.MSELoss(reduction="none")
        components = {"recon": []}

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                rec_loss = criterion(outputs, batch_x)
                components["recon"].append(
                    self._reduce_reconstruction_loss(rec_loss).detach().cpu().numpy()
                )

                quantizer_time, stage_scores = derive_quantizer_time_scores(
                    self.model, rec_loss
                )
                if quantizer_time is not None:
                    components.setdefault("rvq", []).append(
                        quantizer_time.detach().cpu().numpy()
                    )
                if stage_scores:
                    components.setdefault("stage_1", []).append(
                        stage_scores[0].detach().cpu().numpy()
                    )

        score_components = {}
        for key, value in components.items():
            window_scores = np.concatenate(value, axis=0)
            score_components[key] = window_scores_to_sequence(window_scores)
        return score_components

    def _reduce_reconstruction_loss(self, rec_loss: torch.Tensor) -> torch.Tensor:
        return rec_loss.mean(dim=-1)

    def _build_scores(self, train_components, target_components):
        scores = {}
        score_lambda = float(getattr(self.config, "hybrid_score_lambda", 1.0))

        for mode in self.score_modes:
            if mode == "recon":
                scores[mode] = zscore(
                    target_components["recon"],
                    train_components["recon"].mean(),
                    train_components["recon"].std(),
                )
            elif mode == "recon_rvq":
                if "rvq" not in train_components or "rvq" not in target_components:
                    continue
                scores[mode] = zscore(
                    target_components["recon"],
                    train_components["recon"].mean(),
                    train_components["recon"].std(),
                ) + score_lambda * zscore(
                    target_components["rvq"],
                    train_components["rvq"].mean(),
                    train_components["rvq"].std(),
                )
            elif mode == "recon_stage1":
                if "stage_1" not in train_components or "stage_1" not in target_components:
                    continue
                scores[mode] = zscore(
                    target_components["recon"],
                    train_components["recon"].mean(),
                    train_components["recon"].std(),
                ) + score_lambda * zscore(
                    target_components["stage_1"],
                    train_components["stage_1"].mean(),
                    train_components["stage_1"].std(),
                )
            else:
                raise ValueError(f"Unsupported ALV_AD score mode: {mode}")

        if not scores:
            scores["recon"] = target_components["recon"]
        return scores

    def detect_score(self, test: pd.DataFrame):
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        train_components = self._collect_score_components(self.train_eval_loader)
        test_components = self._collect_score_components(self.thre_loader)
        scores = self._build_scores(train_components, test_components)
        primary_score = (
            scores["recon_stage1"] if "recon_stage1" in scores else next(iter(scores.values()))
        )
        return scores, primary_score

    def detect_label(self, test: pd.DataFrame):
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        train_components = self._collect_score_components(self.train_eval_loader)
        test_components = self._collect_score_components(self.test_data_loader)
        thre_components = self._collect_score_components(self.thre_loader)
        train_scores = self._build_scores(train_components, train_components)
        test_scores = self._build_scores(train_components, test_components)
        thre_scores = self._build_scores(train_components, thre_components)

        threshold_scores = test_scores
        pred_scores = thre_scores

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for mode, score in pred_scores.items():
            combined_energy = np.concatenate([train_scores[mode], threshold_scores[mode]], axis=0)
            for ratio in self.config.anomaly_ratio:
                threshold = np.percentile(combined_energy, 100 - float(ratio))
                preds[f"{mode}@{ratio}"] = (score > threshold).astype(int)

        primary_score = (
            pred_scores["recon_stage1"]
            if "recon_stage1" in pred_scores
            else next(iter(pred_scores.values()))
        )
        return preds, primary_score
