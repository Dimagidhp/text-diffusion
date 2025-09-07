import os
import wandb
import matplotlib.pyplot as plt
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
        # Convert timestamp to int for proper comparison and logging
        current_batch = int(state.timestamp.batch)
        if current_batch % self.frequency == 0 and current_batch != 0:
            self.model.noise_schedule.update_optimal_parameters()
            
            # Create directory if it doesn't exist
            os.makedirs('training_figures', exist_ok=True)
            
            # Generate and save the plot locally
            filename = f'training_figures/et_{(self.model.n_parameters/1e6):.2f}M.png'
            self.model.noise_schedule.plot_training_curves(
                f'CrossEntropy-Sigma Curve for {(self.model.n_parameters/1e6):.2f}M parameters every {self.frequency} batches, median={self.model.noise_schedule.medians[-1]}',
                filename=filename
            )
            
            # Log to WandB with proper step conversion
            try:
                # Log the image - use PIL Image or file path
                import PIL.Image
                img = PIL.Image.open(filename)
                logger.log_images({
                    'plots/cross_entropy_sigma_curve': img
                }, step=current_batch)
                
                # Log the metrics
                logger.log_metrics({
                    'noise_schedule/median': float(self.model.noise_schedule.medians[-1]) if self.model.noise_schedule.medians else 0.0,
                    'noise_schedule/mu': float(self.model.noise_schedule.mu),
                    'noise_schedule/sigma': float(self.model.noise_schedule.sigma),
                    'noise_schedule/height': float(self.model.noise_schedule.height),
                    'noise_schedule/offset': float(self.model.noise_schedule.offset)
                }, step=current_batch)
            except Exception as e:
                print(f"Warning: Could not log to WandB: {e}")
                
            clear_output(wait=True)
            

class PlottingData(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_start(self, state: State, logger: Logger):
        # Convert timestamp to int for proper comparison and logging
        current_batch = int(state.timestamp.batch)
        if current_batch % self.frequency == 0 and current_batch != 0:
            # Create directory if it doesn't exist
            os.makedirs('training_figures', exist_ok=True)
            
            # Generate and save the plot locally
            filename = f'training_figures/curves_{(self.model.n_parameters/1e6):.2f}M.png'
            self.model.noise_schedule.plot_entropy_time_curve(
                filename=filename,
                title=f'entropy-time, median={self.model.noise_schedule.medians[-1]}'
            )
            
            # Log to WandB with proper step conversion
            try:
                # Log the image - use PIL Image or file path
                import PIL.Image
                img = PIL.Image.open(filename)
                logger.log_images({
                    'plots/entropy_time_curve': img
                }, step=current_batch)
                
                # Also log the number of completed iterations
                logger.log_metrics({
                    'training/completed_iterations': current_batch,
                    'training/progress_percent': float((current_batch / 15000) * 100)  # Updated for 15000 iterations
                }, step=current_batch)
            except Exception as e:
                print(f"Warning: Could not log to WandB: {e}")


class WriteText(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_start(self, state: State, logger: Logger):
        # Convert timestamp to int for proper comparison and file naming
        current_batch = int(state.timestamp.batch)
        current_epoch = int(state.timestamp.epoch)
        if current_batch % self.frequency == 0 and current_batch != 0:
            self.model.generate_text(16,128,file=f'checkpoints/{(self.model.n_parameters/1e6):.2f}M_ep{current_epoch}_ba{current_batch}.txt')

            

class LRMonitor(Callback):
    def __init__(self,plotting_frequency):
        super().__init__()
        self.plotting_frequency=plotting_frequency

    def batch_end(self, state: State, logger: Logger):
        # Convert timestamp to int for proper comparison and logging
        current_batch = int(state.timestamp.batch)
        if current_batch % self.plotting_frequency == 0 and current_batch != 0:
            assert state.optimizers is not None, 'optimizers must be defined'
            try:
                for optimizer in state.optimizers:
                    lrs = [group['lr'] for group in optimizer.param_groups]
                    name = optimizer.__class__.__name__
                    for idx, lr in enumerate(lrs):
                        logger.log_metrics({f'lr/{name}_group{idx}': float(lr)}, step=current_batch)
            except Exception as e:
                print(f"Warning: Failed to log learning rate: {e}")


class TrainingMonitor(Callback):
    """Additional monitoring for training progress and metrics"""
    def __init__(self, frequency=10):
        super().__init__()
        self.frequency = frequency
        
    def batch_end(self, state: State, logger: Logger):
        # Convert timestamp to int for proper comparison and logging
        current_batch = int(state.timestamp.batch)
        if current_batch % self.frequency == 0:
            # Log training progress metrics with proper conversions
            try:
                current_epoch = int(state.timestamp.epoch)
                current_sample = int(state.timestamp.sample)
                
                logger.log_metrics({
                    'training/batch': current_batch,
                    'training/epoch': current_epoch,
                    'training/samples_processed': current_sample,
                }, step=current_batch)
            except Exception as e:
                print(f"Warning: Could not log training metrics: {e}")


class FindUnused(Algorithm):
    #this class is needed to set find_unused_parameters to True when training with multi-gpu and using self-conditioning
    def match(self, event, state): return False
    def apply(event, state, logger): return None

    @property
    def find_unused_parameters(self): return True