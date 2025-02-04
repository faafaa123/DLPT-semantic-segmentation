# Standard Library imports
import traceback
from functools import reduce
from typing import Callable, Optional

# External imports
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from livelossplot import PlotLosses

# Local imports
from semantic_segmentation.utils import setup_system, create_checkpoint_dir
from semantic_segmentation.configuration import SystemConfig


INPUT_DTYPE = torch.float32
LABEL_DTYPE = torch.long

def make_grid(output_tensor, num_rows):
    """
    Converts a tensor into a grid of images stacked row-wise.

    Args:
        output_tensor (torch.Tensor): Input tensor with shape (batch_size, n_channels, H, W).
        num_rows (int): Number of rows in the grid.

    Returns:
        np.ndarray: Grid of images as a NumPy array.
    """
    plt.figure(figsize=(20, 5))
    output_tensor = output_tensor.cpu().detach()

    batch_size, n_channels, image_height, image_width = output_tensor.shape
    column_images = []
    grid_images = []

    for i in range(n_channels):
        image = output_tensor[0, i]
        column_images.append(image)

        if len(column_images) == num_rows:
            grid_images.append(np.concatenate(column_images, axis=0))
            column_images = []

    return np.concatenate(grid_images, axis=1) if grid_images else np.array([])


def main(
    model,
    optimizer,
    scheduler,
    loss_functions,
    scorer,
    scaler,
    train_dataloader,
    valid_dataloader,
    starting_epoch,
    epochs,
    output_path,
    grad_accum_steps,
    device,
    use_aux=False,
    use_amp=False,
    system_config=SystemConfig(),
):
    setup_system(system_config)
    checkpoint_dir = create_checkpoint_dir(output_path)

    groups = {
        "Score": ["train_score", "valid_score"],
        "Loss (cost function)": ["train_loss", "valid_loss"],
    }

    if len(loss_functions) > 1:
        groups["Train loss components"] = [
            f"train_loss{i+1}" for i in range(len(loss_functions))
        ]
        groups["Valid loss components"] = [
            f"valid_loss{i+1}" for i in range(len(loss_functions))
        ]

    live_plot = PlotLosses(groups=groups)

    H = {
        "train_loss": [],
        "train_score": [],
        "valid_loss": [],
        "valid_score": [],
        "per_class_score": [],
    }
    best_score = 0

    # TODO: set output directory
    writer = SummaryWriter()

    # From:
    # https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
    # https://medium.com/@rekalantar/how-to-visualize-layer-activations-in-pytorch-d0be1076ecc3
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # register forward hooks on the layers of choice
    model.backbone["maxpool"].register_forward_hook(get_activation("maxpool"))
    model.backbone["layer4"][0].conv1.register_forward_hook(get_activation("layer4_0_conv1"))

    try:
        for e in range(starting_epoch, epochs + 1):

            print("\n[INFO] EPOCH: {}/{}".format(e, epochs))

            live_logs = {}

            train_loss, train_loss_components, train_score = train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_functions=loss_functions,
                scorer=scorer,
                dataloader=train_dataloader,
                grad_accum_steps=grad_accum_steps,
                device=device,
                scaler=scaler,
                use_aux=use_aux,
                aux_weight=0.5,
                use_amp=use_amp,
            )

            valid_loss, valid_loss_components, valid_score, mean_per_class_score = (
                validate(model, loss_functions, scorer, valid_dataloader, device)
            )

            # Liveloss tracker
            live_logs["train_loss"] = train_loss
            for i, train_loss_component in enumerate(train_loss_components):
                live_logs[f"train_loss{i+1}"] = train_loss_component
            live_logs["valid_loss"] = valid_loss
            for i, valid_loss_component in enumerate(valid_loss_components):
                live_logs[f"valid_loss{i+1}"] = valid_loss_component
            live_logs["train_score"] = train_score
            live_logs["valid_score"] = valid_score

            # Tensorboard tracker
            writer.add_scalars(
                "Loss",
                {
                    "Train": train_loss,
                    "Valid": valid_loss,
                },
                e,
            )

            writer.add_scalars(
                "Score",
                {
                    "Train": train_score,
                    "Valid": valid_score,
                },
                e,
            )

            # Custom tracker
            H["train_loss"].append(train_loss)
            H["valid_loss"].append(valid_loss)
            H["train_score"].append(train_score)
            H["valid_score"].append(valid_score)
            H["per_class_score"].append(mean_per_class_score)

            print(
                "Epoch train loss: {:.6f} | Epoch train score: {:.4f}".format(
                    train_loss, train_score
                )
            )
            print(
                "Epoch valid loss: {:.6f} | Epoch valid score: {:.4f}".format(
                    valid_loss, valid_score
                )
            )

            if valid_score > best_score:
                best_score = valid_score
                print(f"New best valid score: {best_score:.4f} at epoch {e}")
                output_file_path = checkpoint_dir / f"deeplabv3_best_model.pt"

                # TODO: save train loss history
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": e,  # one-indexed
                    },
                    output_file_path,
                )

            live_plot.update(live_logs)
            live_plot.send()

            # Plot activations to Tensorboard
            grid = make_grid(activations["maxpool"], num_rows=4)
            writer.add_image("maxpool", grid, e, dataformats="HW")

            grid = make_grid(activations["layer4_0_conv1"], num_rows=4)
            writer.add_image("layer4_0_conv1", grid, e, dataformats="HW")
    except KeyboardInterrupt:
        print("Interrupted! Returning output up to this point.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        return H


def calculate_loss_components(
    logits, y_true, loss_functions: list, grad_accum_steps: int, weight=1.0
) -> list[torch.Tensor]:
    loss_components = []
    for loss_fun in loss_functions:
        loss_component = loss_fun(logits, y_true) / grad_accum_steps
        loss_components.append(weight * loss_component)
    return loss_components


def is_batch_update(batch_index, grad_accum_steps, dataloader):
    """
    For Gradient accumulation.
    The model weights are updated only after `grad_accum_steps` steps or at the end of the dataset
    (if it's the final batch).
    """
    return ((batch_index + 1) % grad_accum_steps == 0) or (batch_index + 1 == len(dataloader))      


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_functions: list[Callable],
    scorer,
    dataloader,
    grad_accum_steps: int,
    device: torch.device,
    scaler,
    use_aux: bool = False,
    aux_weight: float = 0.5,
    use_amp=False
) -> tuple[float, list[float], float]:
    """
    Train for one epoch.

    Note:
    All these "batches" should really be called "mini-batches".
    https://www.coursera.org/learn/deep-neural-network/lecture/qcogH?t=332

    For en explanation of gradient accumulation,
    see "MLOps Engineering at Scale - Manning (2022), Ch 8.1.3".

    Also check:
    https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
    https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    

    """

    model.train()

    # Accumulate the total loss and score during the epoch, then compute the average at the end
    epoch_loss_components = []
    epoch_loss = 0
    epoch_score = 0

    num_steps = len(dataloader)
    progress_bar = tqdm(dataloader, total=num_steps)
    for batch_index, (inputs, y_true) in enumerate(progress_bar):

        inputs = inputs.to(device, dtype=INPUT_DTYPE)
        y_true = y_true.to(device, dtype=LABEL_DTYPE)  # Shape (N,H,W)

        with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=use_amp):
            output = model(inputs)

            logits = output["out"]  # Shape (N,C,H,W)
            loss_components = calculate_loss_components(
                logits, y_true, loss_functions, grad_accum_steps
            )
            # TODO: add a loss weight parameter
            loss = reduce(lambda x, y: x + y, loss_components)

            if use_aux and "aux" in output:
                aux_logits = output["aux"]
                aux_loss_components = calculate_loss_components(
                    aux_logits, y_true, loss_functions, grad_accum_steps, weight=aux_weight
                )
                aux_loss = reduce(lambda x, y: x + y, aux_loss_components)
                loss += aux_loss

        # Scales loss and then compute scaled gradients
        scaler.scale(loss).backward()

        # Gradient accumulation.
        # Gradients are calculated for each batch and accumulated over multiple steps.
        # The model weights are updated only after `grad_accum_steps` steps or at the end of the dataset
        # (if it's the final batch).
        if is_batch_update(batch_index, grad_accum_steps, dataloader):
           
            # Apply the accumulated gradients to update the model's weights
            scaler.step(optimizer)

            # Step optimizer learning rate if using the OneCycle policy
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Update the scale for next iteration
            scaler.update()

            # Reset the gradients of all optimized tensors
            optimizer.zero_grad(set_to_none=True)

        # Accumulate total epoch loss
        epoch_loss += loss.item()

        # Append loss components to list
        if use_aux and "aux" in output:
            epoch_loss_components.append(
                [
                    component.item() + aux_comp.item()
                    for component, aux_comp in zip(loss_components, aux_loss_components)
                ]
            )
        else:
            epoch_loss_components.append(
                [component.item() for component in loss_components]
            )

        # Accumulate score
        logits = logits.detach()
        pred_probs = logits.softmax(dim=1)
        max_indices = pred_probs.argmax(dim=1)
        train_score = scorer(max_indices, y_true)
        epoch_score += float(train_score.mean())

        progress_bar.set_description(
            desc=f"Training loss: {loss.item() * grad_accum_steps:.4f} | score: {float(train_score.mean()):.2f}"
        )

    # Average train loss/score during the epoch
    # To obtain the mean loss, the total epoch loss is divided by (num_steps / grad_accum_steps) because:
    #   total_epoch_loss = num_steps * (loss/grad_accum_steps)
    #   loss = (total_epoch_loss/num_steps)*grad_accum_steps
    mean_epoch_loss = epoch_loss / (num_steps / grad_accum_steps)
    mean_epoch_score = epoch_score / num_steps
    mean_epoch_loss_components = (
        np.array(epoch_loss_components).sum(0) / (num_steps / grad_accum_steps)
    ).tolist()

    return mean_epoch_loss, mean_epoch_loss_components, mean_epoch_score


def validate(
    model, loss_functions: list[Callable], scorer, dataloader, device
) -> tuple[float, list[float], float, np.ndarray[np.float32]]:
    """ """

    epoch_loss = 0
    epoch_loss_components = None
    epoch_score = 0
    per_class_score = []

    model.eval()

    # Consider instead using `with torch.inference_mode():`
    # https://pytorch-dev-podcast.simplecast.com/episodes/inference-mode
    with torch.no_grad():

        num_steps = len(dataloader)
        progress_bar = tqdm(dataloader, total=num_steps)
        for inputs, y_true in progress_bar:
            inputs = inputs.to(device, dtype=INPUT_DTYPE)
            y_true = y_true.to(device, dtype=LABEL_DTYPE)

            with torch.autocast(device_type=str(device), dtype=torch.float16):

                logits = model(inputs)["out"]  # Shape (N,C,H,W)
                loss_components: list[torch.Tensor] = []
                # Notice how in validation, the aux branch is not used.
                for i, loss_fun in enumerate(loss_functions):
                    loss_component = loss_fun(logits, y_true)
                    loss_components.append(loss_component)
                loss = reduce(lambda x, y: x + y, loss_components)

            # Accumulate loss
            epoch_loss += loss.item()
            if epoch_loss_components is None:
                epoch_loss_components: list[float] = [0] * len(loss_components)
            for i, component in enumerate(loss_components):
                epoch_loss_components[i] += component.item()

            # Score
            pred_probs = logits.softmax(dim=1)
            max_indices = pred_probs.argmax(dim=1)
            score = scorer(max_indices, y_true)
            epoch_score += float(score.mean())

            # Per-class validation score
            per_class_score.append(score.reshape(1, -1))

            progress_bar.set_description(
                desc=f"Validation loss: {loss.item():.4f} | score: {float(score.mean()):.2f}"
            )

        # Average train loss/score during the epoch
        mean_epoch_loss = epoch_loss / num_steps
        mean_epoch_score = epoch_score / num_steps
        mean_epoch_loss_components = [
            epoch_loss_components[i] / num_steps
            for i in range(len(epoch_loss_components))
        ]

        # Average validation metrics during the epoch
        mean_per_class_score = (
            np.concatenate(per_class_score, axis=0).sum(axis=0) / num_steps
        )

    return (
        mean_epoch_loss,
        mean_epoch_loss_components,
        mean_epoch_score,
        mean_per_class_score,
    )
