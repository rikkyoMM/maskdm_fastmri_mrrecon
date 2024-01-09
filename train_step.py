import os
import math
import wandb
from tqdm.auto import tqdm

import torch
from torchvision import utils

from utils.helper import num_to_groups, has_int_squareroot, exists

# 自分で追加した画像保存コード
# バッチデータを1枚の画像として保存する関数
def save_batch_images(batch_tensor, filename, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Save a batch of images as a grid in a single image file.

    Parameters:
    - batch_tensor (torch.Tensor): The batch of images, expected shape [batch_size, channels, height, width].
    - filename (str): Path and name of the file to save.
    - nrow (int, optional): Number of images in each row of the grid.
    - Other parameters are optional and are passed to `make_grid`.
    """
    # print(len(batch_tensor))
    grid_img = utils.make_grid(batch_tensor[0], nrow=nrow, padding=padding, normalize=normalize, range=value_range, scale_each=scale_each, pad_value=pad_value)
    utils.save_image(grid_img, filename)


def save_training_state(step, model, ema, accelerator, opt, milestone, results_folder="./results"):
    if not accelerator.is_local_main_process:
        return

    data = {
        'step': step,
        'model': accelerator.get_state_dict(model),
        'opt': opt.state_dict(),
        'ema': ema.state_dict(),
        'scaler': accelerator.scaler.state_dict() if exists(accelerator.scaler) else None
    }

    torch.save(data, str(results_folder / f'model-{milestone}.pt'))


def train(
        model,
        loader,
        accelerator,
        opt,
        start_step = 0,
        ema = None,
        gradient_accumulate_every = 1,
        train_num_steps = 100000,
        save_and_sample_every = 1000,
        num_samples = 25,
        batch_size = 5,
        results_folder = './results',
        clip_grad = 1.0,

        lr_schedule_value = None,
    ):
    assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'

    current_step = start_step
    device = accelerator.device

    with tqdm(initial = start_step, total = train_num_steps, disable = not accelerator.is_main_process) as pbar:
        while current_step < train_num_steps:
            
            # learning rate schedule
            if lr_schedule_value is not None:
                for param in opt.param_groups:
                    param["lr"] = lr_schedule_value[current_step]

            total_loss = 0.
            for _ in range(gradient_accumulate_every):
                batch = next(loader)
                # print(current_step)

                # 追加したコード、batch画像の保存
                # save_images(batch['image'], f'batchimage/batch_{_}.png')
                # save_images(batch, f'batchimage/batch_{_}.png')
                # save_batch_images(batch, f'batchimage/batch_{current_step}.png', nrow=8)  # nrowは必要に応じて調整してください
                # print(batch)
                with accelerator.autocast():
                    loss_dict = model(batch)
                    loss = loss_dict["loss"]
                    loss = loss / gradient_accumulate_every
                    total_loss += loss.item()

                if torch.isnan(loss).sum() !=0:
                    print("Nan occurs in loss")

                accelerator.backward(loss)
            
            if clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), clip_grad)
            desc = ""
            for k,v in loss_dict.items():
                if k == "loss": continue
                desc += f'{k}: {v:.4f} '
            pbar.set_description(desc)

            if wandb.run is not None and accelerator.is_main_process:
                wandb.log({"loss":total_loss})
                for k,v in loss_dict.items():
                    if k == "loss": continue
                    wandb.log({k:v})

            accelerator.wait_for_everyone()

            opt.step()
            opt.zero_grad()

            accelerator.wait_for_everyone()

            current_step += 1
            if accelerator.is_main_process:
                ema.to(device)
                ema.update()

                if current_step != 0 and current_step % save_and_sample_every == 0:
                    ema.ema_model.eval()

                    milestone = current_step // save_and_sample_every
                    milestone_folder = results_folder / f"milestone_{milestone}"
                    os.makedirs(milestone_folder, exist_ok=True)

                    batches = num_to_groups(num_samples, batch_size)
                    save_training_state(current_step, model, ema, accelerator, opt, milestone, results_folder=results_folder)

                    with torch.no_grad():
                        all_images_list = list(map(lambda n: ema.ema_model.sample(batch_size=n), batches))
                    all_images = torch.cat(all_images_list, dim=0)
                    
                    utils.save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(num_samples)))

                    # ここで各画像をJPEGとして保存
                    for i, image in enumerate(all_images):
                        utils.save_image(image, milestone_folder / f'sample_{i}.jpg', normalize=True)

            pbar.update(1)

    accelerator.print('training complete')  