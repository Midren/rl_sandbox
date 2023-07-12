from torch.utils.tensorboard.writer import SummaryWriter
import wandb
import typing as t
import omegaconf
from flatten_dict import flatten


class SummaryWriterMock():
    def __init__(self):
        self.log_dir = None

    def add_scalar(*args, **kwargs):
        pass

    def add_video(*args, **kwargs):
        pass

    def add_image(*args, **kwargs):
        pass

    def add_histogram(*args, **kwargs):
        pass

    def add_figure(*args, **kwargs):
        pass

class WandbWriter():
    def __init__(self, project: str, comment: str, cfg: t.Optional[omegaconf.DictConfig]):
        self.run = wandb.init(
            project=project,
            name=comment,
            notes=comment,
            config=flatten(omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), reducer=lambda x, y: f"{x}-{y}" if x is not None else y) if cfg else None
        )
        self.log_dir = wandb.run.dir

    def add_scalar(self, name: str, value: t.Any, global_step: int):
        wandb.log({name: value}, step=global_step)

    def add_image(self, name: str, image: t.Any, global_step: int, dataformats: str = 'CHW'):
        match dataformats:
            case "CHW":
                mode = "RGB"
            case "HW":
                mode = "L"
            case _:
                raise RuntimeError("Not supported dataformat")
        wandb.log({name: wandb.Image(image, mode=mode)}, step=global_step)

    def add_video(self, name: str, video: t.Any, global_step: int, fps: int):
        wandb.log({name: wandb.Video(video[0], fps=fps)}, step=global_step)

    def add_figure(self, name: str, figure: t.Any, global_step: int):
        wandb.log({name: wandb.Image(figure)}, step=global_step)

class Logger:
    def __init__(self, type: t.Optional[str],
                       cfg: t.Optional[omegaconf.DictConfig] = None,
                       project: t.Optional[str] = None,
                       message: t.Optional[str] = None,
                       log_grads: bool = True,
                       log_dir: t.Optional[str] = None
                 ) -> None:
        self.type = type
        msg = message or ""
        match type:
            case "tensorboard":
                self.writer = SummaryWriter(comment=msg, log_dir=log_dir)
            case "wandb":
                self.writer = WandbWriter(project=project, comment=msg, cfg=cfg)
            case None:
                self.writer = SummaryWriterMock()
            case _:
                raise ValueError(f"Unknown logger type: {type}")
        self.log_grads = log_grads


    def log(self, losses: dict[str, t.Any], global_step: int, mode: str = 'train'):
        for loss_name, loss in losses.items():
            if 'grad' in loss_name:
                if self.log_grads:
                    self.writer.add_histogram(f'{mode}/{loss_name}', loss, global_step)
            else:
                self.writer.add_scalar(f'{mode}/{loss_name}', loss.item(), global_step)

    def add_scalar(self, name: str, value: t.Any, global_step: int):
        self.writer.add_scalar(name, value, global_step)

    def add_image(self, name: str, image: t.Any, global_step: int, dataformats: str = 'CHW'):
        self.writer.add_image(name, image, global_step, dataformats=dataformats)

    def add_video(self, name: str, video: t.Any, global_step: int):
        self.writer.add_video(name, video, global_step, fps=20)

    def add_figure(self, name: str, figure: t.Any, global_step: int):
        self.writer.add_figure(name, figure, global_step)

    def log_dir(self) -> str:
        return self.writer.log_dir
