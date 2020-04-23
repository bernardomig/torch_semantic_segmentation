
import os
from ignite.handlers import global_step_from_engine
from ignite.engine import Events
import logging
from logging import info
import wandb


def setup_logging(log_dir, trainer, evaluator,
                  metric_names=['miou', 'accuracy'],
                  freq=100):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    @trainer.on(Events.ITERATION_COMPLETED(every=freq))
    def log_training(engine):
        state = engine.state
        metrics = state.metrics
        step = trainer.state.iteration

        metrics = ', '.join([
            f'{name}={value}' for name, value in metrics.items()])
        info(f'Training results for iteration {step}: {metrics}.')

    @evaluator.on(Events.COMPLETED)
    def log_evaluation(engine):
        state = evaluator.state
        metrics = state.metrics
        step = trainer.state.iteration

        metrics = ', '.join([
            f'{name}={metrics[name]}' for name in metric_names])
        info(f'Evaluation results for iteration {step}: {metrics}.')

    @trainer.on(Events.ITERATION_COMPLETED(every=freq))
    def log_training(engine):
        state = engine.state
        metrics = state.metrics
        step = trainer.state.iteration

        metrics = {
            f'train/{name}': value
            for name, value in metrics.items()
        }
        wandb.log(metrics, step=step)

    @evaluator.on(Events.COMPLETED)
    def log_evaluation(engine):
        state = engine.state
        metrics = state.metrics
        step = trainer.state.iteration

        metrics = {
            f'val/{name}': value
            for name, value in metrics.items()
        }
        wandb.log(metrics, step=step)
