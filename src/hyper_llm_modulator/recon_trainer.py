import logging
import random

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint

logger = logging.getLogger()


def train(
    args,
    hypermod,
    train_data,
    task_embs_dict,
    layer_indices,
    n_batches,
    n_minibatches,
    tasks_per_batch,
    n_embs_per_sampled_task,
    optimizer,
    lr_scheduler,
    device,
    save_dir,
):
    curstep = 1

    for ep in (pbar := tqdm(range(1, args.epochs + 1), total=n_batches)):
        tasks = random.sample(args.train_ds_names, len(args.train_ds_names))
        # avg train unnorm error over all tasks
        err = 0
        losses = []
        grad_norms = []
        for start_idx in range(0, len(tasks), tasks_per_batch):
            # sample tasks (so that we train all the layer/depth embedding for sampled tasks)
            batch_tasks = tasks[start_idx : start_idx + tasks_per_batch]
            batch_data = {task: train_data[task] for task in batch_tasks}
            # sample task embs for each task
            if task_embs_dict is not None:
                for task in batch_data:
                    task_embs = task_embs_dict[task]
                    idx = torch.randperm(len(task_embs))[:n_embs_per_sampled_task]
                    batch_data[task]["task_embs"] = task_embs[idx]
            loss, unnorm_err = compute_loss(args, hypermod, batch_data, layer_indices, device)
            err += unnorm_err
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(hypermod.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)

            pbar.update(1)
            pbar.set_description(f"loss: {loss.item():.4E}")
            if (curstep % args.logging_freq == 0) or (curstep == n_batches):
                keys = ["train/recon_loss", "train/grad_norm", "train/lr"]
                vals = [torch.tensor(losses).mean(), torch.tensor(grad_norms).mean(), lr_scheduler.get_last_lr()[0]]
                for k, v in zip(keys, vals):
                    log_scalar(k, v, curstep)

            curstep += 1
        err = err / n_minibatches
        if ep % args.logging_freq == 0:
            logger.info(f"Epoch {ep}: avg unnorm error: {err:.4E}")
            wandb.log({"train/unnorm_err": err}, step=curstep)

    # save final model
    hypermod.eval()
    sd = hypermod.state_dict()
    save_path = f"{save_dir}/hypermod.pt"
    torch.save(sd, save_path)
    eval_hypermod_checkpoint(save_path, device, curstep, full_eval=True)


def log_scalar(metric_name, val, curstep):
    if wandb.run is not None:
        wandb.log({metric_name: val}, step=curstep)
    logger.info(f"{metric_name}: {val:.4f}")


def compute_loss(args, hypermod, batch_data, layer_indices, device):
    aux_loss = 0
    loss = 0
    err = 0

    tasks = list(batch_data.keys())
    n_tasks = len(batch_data)
    n_embs = len(batch_data[list(batch_data.keys())[0]]["task_embs"])

    # [0 ... 0, 1 ... 1, ..., 32 ... 32]
    # each repeated n_embs * n_tasks times
    repeated_layer_indices = torch.tensor(layer_indices, device=device).repeat_interleave(n_tasks * n_embs)

    # [n_tasks * n_embs, emb_dim]
    task_embs = torch.cat([batch_data[task]["task_embs"] for task in tasks], dim=0)
    emb_dim = task_embs.shape[-1]

    # [n_tasks * n_embs, emb_dim]
    encoder_out = hypermod.task_encoder(task_embs)
    encoded_task_embs = encoder_out["encoded_task_emb"].tile(len(layer_indices), 1)
    if "loss" in encoder_out:
        aux_loss += encoder_out["loss"]

    for target_module in args.target_modules:
        mod = hypermod.get_delta_weights(
            repeated_layer_indices,
            target_module,
            encoded_task_embs,
        )
        target_A = torch.stack([batch_data[task]["lora_A"][target_module] for task in tasks], dim=0)
        target_B = torch.stack([batch_data[task]["lora_B"][target_module] for task in tasks], dim=0)
        if args.factorized:

            # A: [n_layers * n_tasks * n_task_embs, r, in_features]
            # B: [n_layers * n_tasks * n_task_embs, out_features, r]
            A, B = mod
            # A: [n_tasks, n_layers, n_task_embs, r, in_features]
            # B: [n_tasks, n_layers, n_task_embs, out_features, r]
            A = A.view(len(layer_indices), n_tasks, n_embs, *A.shape[1:]).transpose(0, 1)
            B = B.view(len(layer_indices), n_tasks, n_embs, *B.shape[1:]).transpose(0, 1)

            # target_A: [n_tasks, n_layers, r, in_features]
            # target_B: [n_tasks, n_layers, out_features, r]
            with torch.no_grad():
                target_A = target_A.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
                target_B = target_B.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
                if hypermod.pred_z_score:
                    avg_A = hypermod.mean_recon_target["A"][target_module].unsqueeze(0).unsqueeze(2)
                    avg_B = hypermod.mean_recon_target["B"][target_module].unsqueeze(0).unsqueeze(2)
                    std_A = hypermod.std_recon_target["A"][target_module].unsqueeze(0).unsqueeze(2)
                    std_B = hypermod.std_recon_target["B"][target_module].unsqueeze(0).unsqueeze(2)

                    unnorm_A = A.detach() * (std_A + 1e-10) + avg_A
                    unnorm_B = B.detach() * (std_B + 1e-10) + avg_B
                    err += (F.l1_loss(unnorm_A, target_A) + F.l1_loss(unnorm_B, target_B)).detach().item() / 2

                    target_A = (target_A - avg_A) / (std_A + 1e-10)
                    target_B = (target_B - avg_B) / (std_B + 1e-10)

                else:
                    err += (F.l1_loss(A, target_A) + F.l1_loss(B, target_B)).detach().item() / 2
            loss += F.l1_loss(A, target_A) / 2 + F.l1_loss(B, target_B) / 2

        else:
            # deltaW: [n_layers * n_tasks * n_task_embs, out_features, in_features]
            deltaW = mod
            # deltaW: [n_tasks, n_layers, n_task_embs, out_features, in_features]
            deltaW = deltaW.view(len(layer_indices), n_tasks, n_embs, *deltaW.shape[1:]).transpose(0, 1)
            # target_deltaW: [n_tasks, n_layers, out_features, in_features]
            # compute target deltaW from target_A and target_B
            target_deltaW = torch.einsum("ijkl,ijlm->ijkm", target_B, target_A)
            target_deltaW = target_deltaW.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
            l = F.l1_loss(deltaW, target_deltaW * args.delta_w_scaling)
            loss += l
            err += l.item() / args.delta_w_scaling

    loss /= len(args.target_modules)  # average over target modules
    err /= len(args.target_modules)  # average over target modules
    return loss + aux_loss, err
