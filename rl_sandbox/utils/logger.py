from torch.utils.tensorboard.writer import SummaryWriter
import typing as t


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


class Logger:
    def __init__(self, type: t.Optional[str],
                       message: t.Optional[str] = None,
                       log_grads: bool = True,
                       log_dir: t.Optional[str] = None
                 ) -> None:
        self.type = type
        match type:
            case "tensorboard":
                self.writer = SummaryWriter(comment=message or "", log_dir=log_dir)
            case None:
                self.writer = SummaryWriterMock()
            case _:
                raise ValueError(f"Unknown logger type: {type}")
        self.log_grads = log_grads


    def log(self, losses: dict[str, t.Any], global_step: int, mode: str = 'train'):
        for loss_name, loss in losses.items():
            if 'grad' in loss_name:
                if self.log_grads:
                    self.writer.add_histogram(f'train/{loss_name}', loss, global_step)
            else:
                self.writer.add_scalar(f'train/{loss_name}', loss.item(), global_step)

    def add_scalar(self, name: str, value: t.Any, global_step: int):
        self.writer.add_scalar(name, value, global_step)

    def add_image(self, name: str, image: t.Any, global_step: int, dataformats='CHW'):
        self.writer.add_image(name, image, global_step, dataformats=dataformats)

    def add_video(self, name: str, video: t.Any, global_step: int):
        self.writer.add_video(name, video, global_step, fps=20)

    def add_figure(self, name: str, figure: t.Any, global_step: int):
        self.writer.add_figure(name, figure, global_step)

    def log_dir(self) -> str:
        return self.writer.log_dir
