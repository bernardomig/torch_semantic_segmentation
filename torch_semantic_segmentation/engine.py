from functools import partial

from ignite.engine import Engine, Events, _prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss
from ignite.metrics.confusion_matrix import (
    ConfusionMatrix,
    mIoU, IoU,
    DiceCoefficient,
    cmAccuracy
)
from ignite.contrib.handlers import ProgressBar

from apex import amp

__all__ = [
    'create_segmentation_trainer',
    'create_segmentation_evaluator',
]


def create_segmentation_trainer(model, optimizer, loss_fn, device, use_f16=False, logging=True, non_blocking=True):

    def update_fn(_trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if use_f16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        return loss.item()

    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x) \
        .attach(trainer, 'loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_optimizer_params(engine):
        param_groups = optimizer.param_groups[0]
        for h in ['lr', 'momentum', 'weight_decay']:
            if h in param_groups.keys():
                engine.state.metrics[h] = param_groups[h]

    if logging:
        ProgressBar(persist=True) \
            .attach(trainer, ['loss', 'lr'])

    return trainer


def create_segmentation_evaluator(
        model, device,
        num_classes=19,
        loss_fn=None,
        non_blocking=True):

    cm = partial(ConfusionMatrix, num_classes)

    metrics = {
        'iou': IoU(cm()),
        'miou': mIoU(cm()),
        'accuracy': cmAccuracy(cm()),
        'dice': DiceCoefficient(cm()),
    }
    if loss_fn is not None:
        metrics['loss'] = Loss(loss_fn)

    evaluator = create_supervised_evaluator(
        model, metrics, device, non_blocking)

    ProgressBar(persist=False) \
        .attach(evaluator)

    return evaluator
