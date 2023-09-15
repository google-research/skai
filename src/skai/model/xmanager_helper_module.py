"""
Code in this module is adapted from the XManager repo as a temporary workaround for TypeError when importing xm from xmanager in python 3.8.
Code source: https://github.com/deepmind/xmanager/tree/main/xmanager/vizier/vizier_cloud
"""

import re
import time
import abc
from absl import logging
from xmanager.cloud import auth
from typing import Any, Dict, Optional
from google.cloud import aiplatform_v1beta1 as aip

_DEFAULT_LOCATION = 'us-central1'

class StudyFactory(abc.ABC):
  """Abstract class representing vizier study generator."""

  vz_client: aip.VizierServiceClient
  study_config: aip.StudySpec
  num_trials_total: int
  display_name: str

  def __init__(
      self,
      study_config: aip.StudySpec,
      num_trials_total: int,
      display_name: str,
      location: str,
  ) -> None:
    super().__init__()
    self.study_config = study_config
    self.num_trials_total = num_trials_total
    self.display_name = display_name
    self.vz_client = aip.VizierServiceClient(
        client_options=dict(
            api_endpoint=f'{location}-aiplatform.googleapis.com'
        )
    )

  @abc.abstractmethod
  def study(self) -> str:
    raise NotImplementedError


class NewStudy(StudyFactory):
  """Vizier study generator that generates new study from given config."""

  project: str
  location: str

  def __init__(
      self,
      study_config: aip.StudySpec,
      num_trials_total: int = 0,
      display_name: Optional[str] = None,
      project: Optional[str] = None,
      location: Optional[str] = None,
  ) -> None:
    self.project = project or auth.get_project_name()
    self.location = location or _DEFAULT_LOCATION

    super().__init__(
        study_config, num_trials_total, display_name or '', self.location
    )

  def study(self) -> str:
    return self.vz_client.create_study(
        parent=f'projects/{self.project}/locations/{self.location}',
        study=aip.Study(
            display_name=self.display_name, study_spec=self.study_config
        ),
    ).name
  
class VizierController:
  """A Controller that runs Vizier suggested hyperparameters in multiple work units."""

  def __init__(
      self,
      experiment, #: xm.Experiment,
      work_unit_generator, #: Callable[[xm.WorkUnit, Dict[str, Any]], Any],
      vz_client: aip.VizierServiceClient,
      study_name: str,
      num_work_units_total: int,
      num_parallel_work_units: int,
  ) -> None:
    """Create a VizierController.

    Args:
      experiment: XM experiment.
      work_unit_generator: the function that generates WorkUnit from
        hyperparameters.
      vz_client: the Vizier Client used for interacting with Vizier.
      study_name: the study name the controller works on.
      num_work_units_total: number of work units to create in total. (TODO:
        remove this and retrieve from study spec stopping criteria once it is
        settable there.)
      num_parallel_work_units: number of work units to run in parallel.
    """
    self._experiment = experiment
    self._work_unit_generator = work_unit_generator
    self._vz_client = vz_client
    self._study_name = study_name
    self._num_work_units_total = num_work_units_total
    self._num_parallel_work_units = num_parallel_work_units

    self._work_unit_updaters = []

  def run(self, poll_frequency_in_sec: float = 60) -> None:
    """Peridically check and sync status between vizier and work units and create new work units when needed."""
    while True:
      # 1. Complete trial for completed work unit; Early stop first if needed.
      for work_unit_updater in self._work_unit_updaters:
        if not work_unit_updater.completed:
          work_unit_updater.check_for_completion()

      num_exisiting_work_units = len(self._work_unit_updaters)
      num_completed_work_units = sum(
          [wuu.completed for wuu in self._work_unit_updaters]
      )
      if (
          num_exisiting_work_units == self._num_work_units_total
          and num_completed_work_units == self._num_work_units_total
      ):
        print('All done! Exiting VizierController... \n')
        return

      # 3. Get new trials and assign to new work units.
      self._launch_new_work_units()

      time.sleep(poll_frequency_in_sec)

  def _launch_new_work_units(self) -> None:
    """Get hyperparmeter suggestions from Vizier and assign to new work units to run."""
    # 1. Compute num of work units to create next.
    num_existing_work_units = len(self._work_unit_updaters)
    num_running_work_units = len(
        [
            wuu
            for wuu in self._work_unit_updaters
            if wuu.work_unit_status().is_active
        ]
    )
    num_work_units_to_create_total = (
        self._num_work_units_total - num_existing_work_units
    )
    num_work_units_to_create_next = min(
        self._num_parallel_work_units - num_running_work_units,
        num_work_units_to_create_total,
    )

    # 2. Create the work units.
    start_index = num_existing_work_units + 1
    for i in range(start_index, start_index + num_work_units_to_create_next):
      trial = (
          self._vz_client.suggest_trials(
              request=aip.SuggestTrialsRequest(
                  parent=self._study_name,
                  suggestion_count=1,
                  client_id=f'work unit {i}',
              )
          )
          .result()
          .trials[0]
      )
      print(f'Trial for work unit (index: {i}) is retrieved：\n{trial}')

      print(f'Creating work unit (index: {i})... \n')

      def create_gen(index: int, trial: aip.Trial): # -> xm.JobGeneratorType:
        async def gen_work_unit(work_unit, **kwargs): #: xm.WorkUnit, **kwargs):
          await self._work_unit_generator(work_unit, kwargs)

          print(
              f'Work unit (index: {index}, '
              f'id: {work_unit.work_unit_id}) created. \n'
          )
          self._work_unit_updaters.append(
              WorkUnitVizierUpdater(
                  vz_client=self._vz_client, work_unit=work_unit, trial=trial
              )
          )

        return gen_work_unit

      args = {
          'trial_name': trial.name,
          **{p.parameter_id: p.value for p in trial.parameters},
      }
      self._experiment.add(create_gen(i, trial), args)


class WorkUnitVizierUpdater:
  """An updater for syncing completion state between work unit and vizier trial."""

  def __init__(
      self,
      vz_client: aip.VizierServiceClient,
      work_unit, #: xm.WorkUnit,
      trial: aip.Trial,
  ) -> None:
    self.completed = False
    self._vz_client = vz_client
    self._work_unit = work_unit
    self._trial = trial

  def work_unit_status(self): # -> xm.ExperimentUnitStatus:
    return self._work_unit.get_status()

  def check_for_completion(self) -> None:
    """Sync the completion status between WorkUnit and Vizier Trial if needed."""
    if self.completed:
      return

    print(
        'Start completion check for work unit'
        f' {self._work_unit.work_unit_id}.\n'
    )

    if not self.work_unit_status().is_active:
      self._complete_trial(self._trial)
      self.completed = True
    elif (
        self._vz_client.check_trial_early_stopping_state(
            request=aip.CheckTrialEarlyStoppingStateRequest(
                trial_name=self._trial.name
            )
        )
        .result()
        .should_stop
    ):
      print(f'Early stopping work unit {self._work_unit.work_unit_id}.\n')
      self._work_unit.stop()
    else:
      print(f'Work unit {self._work_unit.work_unit_id} is still running.\n')

  def _complete_trial(
      self, trial: aip.Trial, infeasible_reason: Optional[str] = None
  ) -> None:
    """Complete a trial."""
    self._vz_client.complete_trial(
        request=aip.CompleteTrialRequest(
            name=trial.name,
            trial_infeasible=infeasible_reason is not None,
            infeasible_reason=infeasible_reason,
        )
    )
    print(f'Trial {trial.name} is completed\n')

class VizierExploration:
  """An API for launching experiment as a Vizier-based Exploration."""

  def __init__(
      self,
      experiment, #: xm.Experiment,
      job, #: xm.JobType,
      study_factory: StudyFactory,
      num_trials_total: int,
      num_parallel_trial_runs: int,
  ) -> None:
    """Create a VizierExploration.

    Args:
      experiment: the experiment who does the exploration.
      job: a job to run.
      study_factory: the VizierStudyFactory used to create or load the study.
      num_trials_total: total number of trials the experiment want to explore.
      num_parallel_trial_runs: number of parallel runs evaluating the trials.
    """

    async def work_unit_generator(
        work_unit, vizier_params: Dict[str, Any]
    ):
      work_unit.add(job, self._to_job_params(vizier_params))

    if not study_factory.display_name:
      study_factory.display_name = f'X{experiment.experiment_id}'

    self._controller = VizierController(
        experiment,
        work_unit_generator,
        study_factory.vz_client,
        study_factory.study(),
        num_trials_total,
        num_parallel_trial_runs,
    )

  def _to_job_params(self, vizier_params: Dict[str, Any]) -> Dict[str, Any]:
    return {'args': vizier_params}

  def launch(self, **kwargs) -> None:
    self._controller.run(**kwargs)


"""Run Job as a Vizier worker to manager WorkUnit Vizier interaction."""
_TRIAL_NAME_REGEX = (
    r'projects\/[^\/]+\/locations\/[^\/]+\/studies\/[^\/]+\/trials\/[^\/]+'
)


class VizierWorker:
  """Worker that manage interaction between job and Vizier."""

  def __init__(self, trial_name: str) -> None:
    if not re.match(_TRIAL_NAME_REGEX, trial_name):
      raise Exception(
          'The trial_name must be in the form: '
          'projects/{project}/locations/{location}/'
          'studies/{study}/trials/{trial}'
      )

    self._trial_name = trial_name

    location = trial_name.split('/')[3]
    self._vz_client = aip.VizierServiceClient(
        client_options={
            'api_endpoint': f'{location}-aiplatform.googleapis.com',
        }
    )

  def add_trial_measurement(self, step: int, metrics: Dict[str, float]) -> None:
    """Add trial measurements to Vizier."""
    self._vz_client.add_trial_measurement(
        request=aip.AddTrialMeasurementRequest(
            trial_name=self._trial_name,
            measurement=aip.Measurement(
                step_count=step,
                metrics=[
                    aip.Measurement.Metric(metric_id=k, value=v)
                    for k, v in metrics.items()
                ],
            ),
        )
    )
    logging.info('Step %d Metric %s is reported', step, metrics)

  def complete_trial(self, infeasible_reason: Optional[str] = None) -> None:
    """Complete a trial."""
    self._vz_client.complete_trial(
        request=aip.CompleteTrialRequest(
            name=self._trial_name,
            trial_infeasible=infeasible_reason is not None,
            infeasible_reason=infeasible_reason,
        )
    )
    logging.info('Trial %s is completed', self._trial_name)