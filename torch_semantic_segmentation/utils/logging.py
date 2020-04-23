
from ignite.handlers import global_step_from_engine
from ignite.engine import Events
from logging import info


def setup_valiation_logging(evaluator, global_step_transform, metric_names=['miou', 'acc', 'lr']):
    @evaluator.on(Events.COMPLETED)
    def log_metrics(engine):
        state = evaluator.state
        metrics = state.metrics
        step = global_step_transform()

        metrics = ', '.join([
            f'{name}={metrics[name]}' for name in metric_names])
        info(f'Evaluation results for iteration {step}: {metrics}.')


def setup_wandb_logging(trainer, evaluator, freq=100):
    import wandb

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
