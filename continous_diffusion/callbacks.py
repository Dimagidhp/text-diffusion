from composer import Callback, State, Logger, Algorithm
from IPython.display import clear_output
from .utils import median
from .diffusion import Diffusion

class SchedulerUpdater(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            self.model.noise_schedule.update_optimal_parameters()
            self.model.noise_schedule.plot_training_curves(
                f'CrossEntropy-Sigma Curve for {(self.model.n_parameters/1e6):.2f}M parameters every {self.frequency} batches, median={self.model.noise_schedule.medians[-1]}',
                filename=f'training_figures/et_{(self.model.n_parameters/1e6):.2f}M.png'
                )
            clear_output(wait=True)
            

class PlottingData(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_start(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            self.model.noise_schedule.plot_entropy_time_curve(
                filename = f'training_figures/curves_{(self.model.n_parameters/1e6):.2f}M.png',
                title=f'entropy-time, median={self.model.noise_schedule.medians[-1]}'
                )


class WriteText(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_start(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            self.model.generate_text(16,128,file=f'checkpoints/{(self.model.n_parameters/1e6):.2f}M_ep{state.timestamp.epoch}_ba{state.timestamp.batch}.txt')

            

class LRMonitor(Callback):
    def __init__(self,plotting_frequency):
        super().__init__()
        self.plotting_frequency=plotting_frequency

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch%self.plotting_frequency==0 and state.timestamp.batch!=0:
            assert state.optimizers is not None, 'optimizers must be defined'
            for optimizer in state.optimizers:
                lrs = [group['lr'] for group in optimizer.param_groups]
                name = optimizer.__class__.__name__
                for idx, lr in enumerate(lrs):
                    print({f'lr-{name}/group{idx}': lr})


class FindUnused(Algorithm):
    #this class is needed to set find_unused_parameters to True when training with multi-gpu and using self-conditioning
    def match(self, event, state): return False
    def apply(event, state, logger): return None

    @property
    def find_unused_parameters(self): return True