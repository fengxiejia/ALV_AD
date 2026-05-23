import copy

import numpy as np


def adjust_learning_rate(optimizer, epoch: int, args, scheduler=None, printout: bool = True) -> None:
    lr = getattr(args, "learning_rate", getattr(args, "lr", None))
    if lr is None:
        return

    lradj = getattr(args, "lradj", "type1")
    if lradj == "type1":
        lr_adjust = {epoch: lr * (0.5 ** max(epoch - 1, 0))}
    elif lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif lradj == "type3":
        lr_adjust = {epoch: lr if epoch < 3 else lr * (0.9 ** (epoch - 3))}
    elif lradj == "type4":
        lr_adjust = {epoch: lr if epoch < 20 else lr * (0.5 ** (epoch // 20))}
    elif lradj == "type5":
        lr_adjust = {epoch: lr if epoch < 10 else lr * (0.5 ** (epoch // 10))}
    elif lradj == "type6":
        lr_adjust = {
            8: lr * 0.01,
            20: lr * 0.5,
            40: lr * 0.01,
            60: lr * 0.01,
            100: lr * 0.01,
        }
    elif lradj == "constant":
        lr_adjust = {epoch: lr}
    elif lradj == "3":
        lr_adjust = {epoch: lr if epoch < 10 else lr * 0.1}
    elif lradj == "4":
        lr_adjust = {epoch: lr if epoch < 15 else lr * 0.1}
    elif lradj == "5":
        lr_adjust = {epoch: lr if epoch < 25 else lr * 0.1}
    elif lradj == "6":
        lr_adjust = {epoch: lr if epoch < 5 else lr * 0.1}
    elif lradj == "TST" and scheduler is not None:
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        return

    if epoch in lr_adjust:
        updated_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = updated_lr
        if printout:
            print("Updating learning rate to {}".format(updated_lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.check_point = None

    def __call__(self, val_loss, model):
        if not np.isfinite(val_loss):
            self.counter += 1
            if self.verbose:
                print(
                    f"Non-finite validation loss ({val_loss}); "
                    f"skip checkpoint. EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        self.val_loss_min = val_loss
