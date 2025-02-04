from torch.optim import lr_scheduler


def get_scheduler(scheduler_name, optimizer, total_steps, max_lr=None, min_lr=None):
    """ """

    if scheduler_name is None:
        print("No scheduler will be used.")
        return None

    if scheduler_name == "constant":
        start_factor = 1
        print(f"LinearLR scheduler with start_factor={start_factor}.")
        return lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=total_steps
        )

    elif scheduler_name == "onecycle":
        assert max_lr is not None
        print(f"OneCycleLR scheduler with max_lr={max_lr}.")
        return lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )

    elif scheduler_name == "cosine":
        assert min_lr is not None
        T_max = total_steps / 10
        print(f"CosineAnnealingLR scheduler with T_max={max_lr} and eta_min={min_lr}.")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
