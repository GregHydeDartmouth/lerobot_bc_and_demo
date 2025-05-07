
import gc
import torch
import random
import logging
import pandas as pd
from pathlib import Path
import torch as th
from torch.amp import GradScaler
from contextlib import nullcontext
from lerobot.common.envs import PushtEnv
from lerobot.scripts.eval import eval_policy
from lerobot.scripts.train import update_policy
from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs.eval import EvalConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.envs.factory import make_env
from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker

from lerobot.common.policies.bc.configuration_bc import BCConfig
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig


init_logging()

methods = ["bc", "act", "diffusion","vqbet"]

num_trials = 10

for method in methods:
    for trial_num in range(0, num_trials):
        trial_evals = dict()

        seed = random.randint(0, 10_000)
        output_directory = Path(f"./data/log_{method}/trial_{trial_num}")

        if method == "bc":
            pcfg = BCConfig(device='cuda')
        elif method == "act":
            pcfg = ACTConfig(device='cuda')
        elif method == "diffusion":
            pcfg = DiffusionConfig(device='cuda')
        elif method == "vqbet":
            pcfg = VQBeTConfig(device='cuda')

        # prep relevant configs
        dcfg = DatasetConfig(repo_id='lerobot/pusht')
        pt_env = PushtEnv()
        ecfg = EvalConfig(n_episodes=50, batch_size=50, use_async_envs=False)
        cfg = TrainPipelineConfig(
            dcfg,
            env=pt_env,
            policy=pcfg,
            eval=ecfg,
            output_dir=output_directory,
            seed=seed,
            num_workers=4,
            batch_size=64,
            steps=100_000,
            eval_freq=1_000,
            log_freq=500,
            save_freq=100_000,
        )
        cfg.validate()

        # Check device is available
        device = get_safe_torch_device(cfg.policy.device, log=True)

        # make dataset
        dataset = make_dataset(cfg)
        # create dataloader for offline training
        dataloader = th.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True,
            sampler=None,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )

        # make eval env
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

        # construct policy
        policy = make_policy(
                cfg=cfg.policy,
                ds_meta=dataset.meta,
            )
        policy.train()

        # training schedulur and optimizer
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0
        # train and eval metrics
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        train_tracker = MetricsTracker(
            cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
        )

        eval_metrics = {
            "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
            "pc_success": AverageMeter("success", ":.1f"),
            "eval_s": AverageMeter("eval_s", ":.3f"),
        }

        dl_iter = cycle(dataloader)    
        for _ in range(step, cfg.steps):
            batch = next(dl_iter)

            for key in batch:
                if isinstance(batch[key], th.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
                    
            train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    grad_scaler=grad_scaler,
                    lr_scheduler=lr_scheduler,
                    use_amp=cfg.policy.use_amp,
                )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                logging.info(train_tracker)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)


            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    th.no_grad(),
                    th.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                eval_tracker = eval_tracker.to_dict()
                step_ctr = eval_tracker.pop('steps')
                trial_evals[step_ctr] = eval_tracker

        if eval_env:
            eval_env.close()
        logging.info("End of training")
        df = pd.DataFrame.from_dict(trial_evals, orient='index')

        # Save to CSV
        csv_path = output_directory / "eval_metrics.csv"
        df.to_csv(csv_path)
