from blocks.extensions import SimpleExtension
from blocks.dump import MainLoopDumpManager

class MyLearningRateSchedule(SimpleExtension):
	"""My learning rate schedule.

	Divides learning rate by factor if validation cost increases. 
	"""

	def __init__(self, channel, factor, **kwargs):
		kwargs.setdefault("after_every_epoch", True)
		super(MyLearningRateSchedule, self).__init__(**kwargs)
		self.channel = channel
		self.factor = factor
		self.previous = float('inf')

	def do(self, which_callback, *args):
		log = self.main_loop.log
		current = log.current_row[self.channel]
		if current > self.previous:
			learning_rate = self.main_loop.algorithm.step_rule.learning_rate
			new_learning_rate = learning_rate.get_value()/self.factor
			learning_rate.set_value(new_learning_rate)
			setattr(log.current_row, 'learning_rate', new_learning_rate)
		self.previous = current

SAVED_TO = "saved_to"

class EarlyStoppingDump(SimpleExtension):
	"""Saves parameters of best validation score. 

	Makes a `SAVED_TO` record in the log with the dumping destination
    in the case of success and ``None`` in the case of failure.

    Parameters
    ----------
    state_path : str
        The folder to dump the state to. Will be created if it does not
        exist.
	"""

	def __init__(self, state_path, channel, **kwargs):
		kwargs.setdefault("after_every_epoch", True)
		super(EarlyStoppingDump, self).__init__(**kwargs)
		self.state_path = state_path
		self.channel = channel
		self.manager = MainLoopDumpManager(state_path)
		self.best = float('inf')

	def do(self, which_callback, *args):
		log = self.main_loop.log
		current = log.current_row[self.channel]
		try:
			if current < self.best:
				self.manager.dump_parameters(self.main_loop)

			self.manager.dump_iteration_state(self.main_loop)
			self.manager.dump_log(self.main_loop)
			log.current_row[SAVED_TO] = self.manager.folder
		except Exception:
			log.current_row[SAVED_TO] = None
			raise
