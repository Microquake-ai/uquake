from pathlib import Path
import pickle
from uquake.core.inventory import read_inventory
from uquake.core.logging import logger
from uquake.nlloc import (Srces, Control, GeographicTransformation,
                          LocSearchOctTree, LocationMethod,
                          GaussianModelErrors, LocQual2Err, LocSearchGrid,
                          LocSearchMetropolis, LocGrid, Observations,
                          read_hypocenter_file, read_scatter_file,
                          NllocInputFiles)
from uquake.core.event import ConfidenceEllipsoid, OriginUncertainty
from uquake.grid import nlloc as nlloc_grid
from uuid import uuid4
from time import time
import numpy as np
import os
import shutil
from .settings import Settings


def calculate_uncertainty(point_cloud):
    """
    :param point_cloud: location point cloud
    :return: obspy.core.event.OriginUncertainty()
    """

    v, u = np.linalg.eig(np.cov(point_cloud.T))

    major_axis_index = np.argmax(v)

    uncertainty = np.sort(np.sqrt(v))[-1::-1]

    h = np.linalg.norm(u[major_axis_index, :-1])
    vert = u[major_axis_index, -1]

    major_axis_plunge = np.arctan2(-vert, h)
    x = u[major_axis_index, 0]
    y = u[major_axis_index, 1]
    major_axis_azimuth = np.arctan2(x, y)
    major_axis_rotation = 0

    ce = ConfidenceEllipsoid(semi_major_axis_length=uncertainty[0],
                             semi_intermediate_axis_length=uncertainty[1],
                             semi_minor_axis_length=uncertainty[2],
                             major_axis_plunge=major_axis_plunge,
                             major_axis_azimuth=major_axis_azimuth,
                             major_axis_rotation=major_axis_rotation)

    return OriginUncertainty(confidence_ellipsoid=ce)


class ProjectManager(object):

    inventory_file_name = 'inventory.xml'

    def __init__(self, path, project_name, network_code, use_srces=False):
        """
        Interface to manage project providing an interface to selected
        components of the NonLinLoc software by Anthony Lomax.

        :param path: base path
        :type path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        :program use_srces: if True use the srces files instead of the the
        inventory file should both files be present (default=False)
        :Example:
        >>> from uquake.grid import nlloc as nlloc_grid
        >>> from uquake.nlloc import nlloc
        # initialize the project with the root path of the project,
        the project and the network names
        >>> project_name = 'test'
        >>> network_code = 'test'
        >>> root_path=''
        >>> pm = nlloc.ProjectManager(root_path, project_name, network_code)
        # this will initialize the project and create a specific directory
        # structure if the directories do not exist.

        # add an inventory file to the project
        >>> path_to_uquake_inventory_file = 'PATH/TO/THE/INVENTORY/FILE.xml'

        ..note :: the uquake inventory object inherit from the obspy inventory
                  and
                  behaves in a very similar way. It, however, differs from
                  its parent has it implements properties specific to local
                  earthquake who are either expressed using UTM coordinates
                  system or a local coordinates system that cannot necessarily
                  be translated to latitude and longitude. Beyond this minor
                  difference, uquake inventory structure also differs from
                  the standard inventory and implements a slightly different
                  hierarchy more representative of mine monitoring systems.
                  The uquake Inventory object hierarchy is as follows:
                 1. Inventory
                    1.1. Networks: A seismic network. a network includes at
                                   least one data acquisition station.
                        1.1.1 Stations: A place where the data acquisition is
                                        performed. For instance, station
                                        usually includes power, communication
                                        and data acquisition equipment. One
                                        or more sensor can be connected to a
                                        station.
                            1.1.1.1 Sensors: A instrument converting a
                                             physical phenomenon to data either
                                             digital or analog. A sensor
                                             comprises one or more channel.
                                1.1.1.1.1 Channels: A channel is a the smallest
                                                    unit of measuring.


        >>> inventory = read_inventory(path_to_uquake_inventory_file)
        >>> pm.add_inventory(inventory)

        # alternatively, sensors can be added to the using the srces object.
        # Sensors can be added this way from a nlloc.nlloc.Srces object using
        the .add_srces method.
        ..note:: srces stands for sources and it is the nomenclature used in
                 NonLinLoc. This might be a soure of confusion for the users.
                 In addition, what NonLinLoc refers to as station is called a
                 sensors in this context.

        # srces object can be constructed from an inventory as follows:
        >>> srces = Srces.from_inventory(inventory)
        # alternatively, each sensors can be added individually using the
        .add_sensor method. As follows:
        >>> srces = Srces()
        >>> x = 250
        >>> y = 250
        >>> z = 250
        >>> elevation = 0
        >>> srces.add_sensor('sensor label', x, y, z, elev=elevation)

        # Srces can be added to the project as follows:
        >>> pm.add_srces(srces)

        # add the velocity models to the project. P- and S-wave velocity models
        can be added separately from a nlloc.grid.VelocityGrid3D
        using the .add_velocity method or from a nlloc.grid.VelocityEnsemble
        object using the .add_velocities method

        >>> origin = [0, 0, 0]
        >>> spacing = [25, 25, 25]
        >>> dimensions = [100, 100, 100]

        >>> nlloc_grid.VelocityGrid3D()
        >>> vp = 5000  # P-wave velocity in m/s
        >>> vs = 3500  # S-wave velocity in m/s
        >>> p_velocity = nlloc_grid.VelocityGrid3D(network_code, dimensions,
        >>>                                        origin, spacing, phase='P',
        >>>                                        value=5000)
        >>> s_velocity = nlloc_grid.VelocityGrid3D(network_code, dimensions,
        >>>                                        origin, spacing, phase='S',
        >>>                                        value=5000 )
        >>> pm.add_velocity(p_velocity)
        >>> pm.add_velocity(s_velocity)
        # Alternatively
        >>> velocities = nlloc_grid.VelocityGridEnsemble(p_velocity,
        >>>                                              s_velocity)
        >>> pm.add_velocities(velocities)
        # Adding a velocity model of the velocity models triggers the
        # calculation of the travel time grids.
        # It is possible to manually trigger the calculation of the travel
        # time grid by invoking
        >>> pm.init_travel_time_grid()
        # this should not, however, be required.

        # prior to running the location, NonLinLoc need to be configured.
        # configuring NonLinLoc can be done using the nlloc.nlloc.NonLinLoc
        # object
        >>> nonlinloc = nlloc.NonLinLoc()
        # this will initialize the nonlinloc object sith default value. Those
        # value have been used to locate seismic events in a volumes of
        # approximately 3000 m x 3000 m x 1500 m. The parameters should be
        # provide adequate results for volumes of similar scale but would need
        # to be adapted to smaller or larger volumes.

        """

        self.project_name = project_name
        self.network_code = network_code
        self.root_directory = Path(path) / project_name / network_code
        # create the directory if it does not exist
        self.root_directory.mkdir(parents=True, exist_ok=True)

        self.inventory_location = self.root_directory / 'inventory'
        self.inventory_location.mkdir(parents=True, exist_ok=True)
        self.inventory_file = self.inventory_location / 'inventory.xml'
        self.srces_file = self.inventory_location / 'srces.pickle'

        self.settings_location = self.root_directory / 'settings'

        self.settings_location.mkdir(parents=True, exist_ok=True)

        self.settings_file = self.settings_location / 'settings.toml'

        if not self.settings_file.exists():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                    'settings_template.toml'

            shutil.copyfile(settings_template, self.settings_file)

        self.settings = Settings(self.settings_file)

        self.srces = None
        self.inventory = None

        if (not self.inventory_file.exists()) and \
                (not self.srces_file.exists()):
            logger.warning('the project does not contain an inventory file nor'
                           'an srces file. to add an inventory file use '
                           'the add_inventory method. Alternatively, sensors '
                           'information can be provided using an Srces object.'
                           'Note, however, that an Srces object only contains'
                           'the sensors location information. When '
                           'the inventory file is present, the Srces object '
                           'is automatically constructed from the inventory.'
                           'A Srces object can be added using the add_srces '
                           'method. If both the Srces and inventory files are'
                           'present in the project directory, the inventory'
                           'file takes precedence and the srces file is'
                           'ignored unless')

        elif self.srces_file.exists() and use_srces:
            with open(self.srces_file, 'rb') as srces_file:
                logger.info('srces will be read from the file and not build'
                            'from the inventory file')
                self.srces = pickle.load(srces_file)

            if self.inventory_file.exists():
                self.inventory = read_inventory(str(self.inventory_file))

        elif self.inventory_file.exists():
            self.inventory = read_inventory(str(self.inventory_file))
            self.srces = Srces.from_inventory(self.inventory)
            logger.info('srces will be build from the inventory file. The '
                        'srces.pickle file will be replaced.')
            with open(self.srces_file, 'wb') as srces_file:
                pickle.dump(self.srces, srces_file)

        elif self.srces_file.exists():
            logger.info('no inventory file in the project. srces file will be '
                        'used instead.')
            with open(self.srces_file, 'rb') as srces_file:
                self.srces = pickle.load(srces_file)

        self.velocity_grid_location = self.root_directory / 'velocities'
        self.velocity_grid_location.mkdir(parents=True, exist_ok=True)

        p_vel_base_name = nlloc_grid.VelocityGrid3D.get_base_name(network_code, 'P')
        self.p_velocity_file = self.velocity_grid_location / p_vel_base_name
        p_files = self.velocity_grid_location.glob(f'{p_vel_base_name}*')
        if len(list(p_files)) == 0:
            logger.warning('the project does not contain a p-wave velocity '
                           'model. to add a p-wave velocity model to the '
                           'project please use the add_velocity or '
                           'add_velocities methods.')
            self.p_velocity = None

        else:
            self.p_velocity = nlloc_grid.read_grid(p_vel_base_name,
                                                   path=str(
                                                 self.velocity_grid_location))

        s_vel_base_name = nlloc_grid.VelocityGrid3D.get_base_name(self.network_code,
                                                            'S')
        self.s_velocity_file = self.velocity_grid_location / s_vel_base_name
        s_files = self.velocity_grid_location.glob(f'{s_vel_base_name}*')
        if len(list(s_files)) == 0:
            logger.warning('the project does not contain a s-wave velocity '
                           'model. to add a s-wave velocity model to the '
                           'project please use the add_velocity or '
                           'add_velocities methods.')
            self.s_velocity = None
        else:
            self.s_velocity = nlloc_grid.read_grid(s_vel_base_name,
                                                   path=str(
                                                 self.velocity_grid_location))

        self.velocities = None
        if (self.p_velocity is not None) and (self.s_velocity is not None):
            self.velocities = nlloc_grid.VelocityGridEnsemble(self.p_velocity,
                                                              self.s_velocity)

        self.travel_time_grid_location = self.root_directory / 'times'
        self.travel_time_grid_location.mkdir(parents=True, exist_ok=True)
        file_list = list(self.travel_time_grid_location.glob('*time*'))
        if len(file_list) == 0:
            logger.warning('the project does not contain travel time grids. '
                           'to initialize the travel-time grid use the '
                           'init_travel_time_grid method. Note that '
                           'this require the project to contain both '
                           'an inventory and a velocities files.')

            self.travel_times = None

        else:
            self.travel_times = nlloc_grid.TravelTimeEnsemble.from_files(
                self.travel_time_grid_location)

        self.run_id = str(uuid4())
        self.current_run_directory = self.root_directory / 'run' / self.run_id
        self.current_run_directory.mkdir(parents=True, exist_ok=False)

        self.output_file_path = self.current_run_directory / 'outputs'
        self.output_file_path.mkdir(parents=True, exist_ok=True)

        self.observation_path = self.current_run_directory / 'observations'
        self.observation_path.mkdir(parents=True, exist_ok=True)
        self.observation_file_name = 'observations.obs'
        self.observation_file = self.observation_path / \
                                self.observation_file_name
        self.observations = None

        self.template_directory = self.root_directory / 'templates'

        self.template_directory.mkdir(parents=True, exist_ok=True)

        self.template_ctrl_file = self.template_directory / \
                                  'ctrl_template.pickle'

        if self.template_ctrl_file.exists():
            with open(self.template_ctrl_file, 'rb') as template_ctrl:
                self.control_template = pickle.load(template_ctrl)
        else:
            self.control_template = None
            self.add_template_control()

        self.control_file = self.current_run_directory / 'run.nll'

        self.last_event_hypocenter = None
        self.last_event_time = None

        self.last_location = None

    @classmethod
    def init_from_settings(cls, settings):
        try:
            base_directory = settings.project.base_directory
            name = settings.project.name
            network = settings.project.network
        except Exception as e:
            logger.error(e)
            logger.error('a settings_template.toml file comprising the three'
                         ' following variable must be present either in '
                         'the current directory of in a directory defined '
                         'by the UQUAKE_CONFIG environment variable. '
                         'Alternatively, the following environment variables '
                         'shall be set: \n'
                         '- UQUAKE_PROJECT__BASE_DIRECTORY\n'
                         '- UQUAKE_PROJECT__NAME\n'
                         '- UQUAKE_PROJECT__NETWORK')
        return cls(base_directory, name, network)

    def init_travel_time_grid(self):
        """
        initialize the travel time grids
        """
        logger.info('initializing the travel time grids')
        t0 = time()
        seeds = self.srces.locs
        seed_labels = self.srces.labels

        if self.srces is None:
            return

        if self.p_velocity:
            tt_gs_p = self.p_velocity.to_time_multi_threaded(seeds,
                                                             seed_labels)

        if self.s_velocity:
            tt_gs_s = self.s_velocity.to_time_multi_threaded(seeds,
                                                             seed_labels)

        if self.p_velocity and self.s_velocity:
            self.travel_times = tt_gs_p + tt_gs_s

        elif self.p_velocity:
            logger.warning('s-wave velocity model is not set, travel-times '
                           'will'
                           'will be generated for ')
            self.travel_times = tt_gs_p

        elif self.s_velocity:
            self.travel_times = tt_gs_s

        else:
            return

        # cleaning the directory before writing the new files

        for fle in self.travel_time_grid_location.glob('*time*'):
            fle.unlink(missing_ok=True)

        self.travel_times.write(self.travel_time_grid_location)
        t1 = time()
        logger.info(f'done initializing the travel time grids in '
                    f'{t1 - t0:0.2f} seconds')

    def add_inventory(self, inventory, create_srces_file: bool = True,
                      initialize_travel_time: bool = True):
        """
        adding a inventory object to the project
        :param inventory: station xml inventory object
        :type inventory: uquake.core.inventory.Inventory
        :param create_srces_file: if True create or replace the srces file
        (Default: True)
        :param initialize_travel_time: if True the travel time grids are
        initialized (default: True)
        :return:
        """

        inventory.write(str(self.inventory_file))
        self.srces = Srces.from_inventory(inventory)
        if create_srces_file:
            with open(self.srces_file, 'wb') as srces_file:
                pickle.dump(self.srces, srces_file)

        if initialize_travel_time:
            self.init_travel_time_grid()
        else:
            logger.warning('the travel time grids will not be initialized, '
                           'the inventory and the travel time grids might '
                           'be out of sync. To initialize the travel time '
                           'grids make sure initialize_travel_time is set to '
                           'True')

    def add_srces(self, srces, force=False, initialize_travel_time=True):
        """
        add a list of sources to the projects
        :param srces: list of sources or sensors
        :param force: force the insertion of the srces object if an inventory
        file is present
        :param initialize_travel_time: if True, initialize the travel time
        grid
        :type srces: Srces

        ..warning:: travel time should be initialized when the sensors/srces
        are updated. Not doing so, may cause the sensors/source and the
        travel time grids to be incompatible.
        """

        if not isinstance(srces, Srces):
            raise TypeError(f'Expecting type {Srces}, given '
                            f'{type(srces)}')

        if self.inventory is not None:
            logger.warning('The project already has an inventory file!')
            if not force:
                logger.warning('exiting...')
                return
            else:
                logger.warning('the force flag value is True, srces object '
                               'will be added.')

        self.srces = srces
        with open(self.srces_file, 'wb') as srces_file:
            pickle.dump(self.srces, srces_file)

        if initialize_travel_time:
            self.init_travel_time_grid()
        else:
            logger.warning('the travel time grids will not be initialized, '
                           'the inventory and the travel time grids might '
                           'be out of sync. To initialize the travel time '
                           'grids make sure initialize_travel_time is set to '
                           'True')

        self.init_travel_time_grid()

    def add_velocities(self, velocities, initialize_travel_time=True):
        """
        add P- and S-wave velocity models to the project
        :param velocities: velocity models
        :type velocities: nlloc.grid.VelocityEnsemble
        :param initialize_travel_time: if True, initialize the travel time
        grid

        ..warning:: travel time should be initialized when the sensors/srces
        are updated. Not doing so, may cause the sensors/source and the
        travel time grids to be incompatible.
        """

        # velocities.write(path=self.velocity_grid_location)

        self.velocities = velocities

        for key in velocities.keys():
            self.add_velocity(velocities[key],
                              initialize_travel_time=False)

        if initialize_travel_time:
            self.init_travel_time_grid()
        else:
            logger.warning('the travel time grids will not be initialized, '
                           'the inventory and the travel time grids might '
                           'be out of sync. To initialize the travel time '
                           'grids make sure initialize_travel_time is set to '
                           'True')

    def add_velocity(self, velocity, initialize_travel_time=True):
        """
        add P- or S-wave velocity model to the project
        :param velocity: p-wave velocity model
        :type velocity: nlloc.grid.VelocityGrid3D
        :param initialize_travel_time: if true initialize the travel time grids

        ..warning:: travel time should be initialized when the sensors/srces
        are updated. Not doing so, may cause the sensors/source and the
        travel time grids to be incompatible.
        """

        velocity.write(path=self.velocity_grid_location)

        if velocity.phase.upper() == 'P':
            self.p_velocity = velocity

        else:
            self.s_velocity = velocity

        if initialize_travel_time:
            self.init_travel_time_grid()
        else:
            logger.warning('the inventory and the travel time grids might'
                           'be out of sync.')

    def clean_run(self):
        """
        remove the files and directory related to a particular run id and
        create a new run id and related directory.
        """
        logger.info(f'removing {self.current_run_directory}')
        for item in list(self.current_run_directory.glob('*')):
            item.unlink()
        self.current_run_directory.rmdir()
        self.run_id = str(uuid4())
        self.current_run_directory = self.root_directory / 'run' / self.run_id

    def clean_project(self):
        pass

    def add_template_control(self, control=Control(message_flag=1),
                             transformation=GeographicTransformation(),
                             locsig=None, loccom=None,
                             locsearch=LocSearchOctTree.init_default(),
                             locmeth=LocationMethod.init_default(),
                             locgau=GaussianModelErrors.init_default(),
                             locqual2err=LocQual2Err(0.0001, 0.0001, 0.0001,
                                                     0.0001, 0.0001),
                             **kwargs):

        if not isinstance(control, Control):
            raise TypeError(f'control is type {type(control)}. '
                            f'control must be type {Control}.')

        if not issubclass(type(transformation), GeographicTransformation):
            raise TypeError(f'transformation is type {type(transformation)}. '
                            f'expecting type'
                           f'{GeographicTransformation}.')

        if not locsearch.type == 'LOCSEARCH':
            raise TypeError(f'locsearch is type {type(locsearch)}'
                            f'expecting type '
                            f'{LocSearchGrid} '
                            f'{LocSearchMetropolis}, or '
                            f'{LocSearchOctTree}.')

        if not isinstance(locmeth, LocationMethod):
            raise TypeError(f'locmeth is type {type(locmeth)}, '
                            f'expecting type {LocationMethod}')

        if not isinstance(locgau, GaussianModelErrors):
            raise TypeError(f'locgau is type {type(locgau)}, '
                            f'expecting type {GaussianModelErrors}')

        if not isinstance(locqual2err, LocQual2Err):
            raise TypeError(f'locqual2err is type {type(locqual2err)}, '
                            f'expecting type {LocQual2Err}')

        dict_out = {'control': control,
                    'transformation': transformation,
                    'locsig': locsig,
                    'loccom': loccom,
                    'locsearch': locsearch,
                    'locmeth': locmeth,
                    'locgau': locgau,
                    'locqual2err': locqual2err}

        with open(self.template_ctrl_file, 'wb') as template_ctrl:
            pickle.dump(dict_out, template_ctrl)

        self.control_template = dict_out

    def write_control_file(self):
        with open(self.control_file, 'w') as control_file:
            control_file.write(self.control)

    def add_observations(self, observations):
        """
        adding observations to the project
        :param observations: Observations
        """
        if not isinstance(observations, Observations):
            raise TypeError(f'observations is type {type(observations)}. '
                            f'observations must be type {Observations}.')
        self.observation_path.mkdir(parents=True, exist_ok=True)
        observations.write(self.observation_file_name,
                           path=self.observation_path)

        self.observations = observations

    def run_location(self, observations=None, calculate_rays=True,
                     delete_output_files=True, event=None):

        import subprocess

        if event is not None:
            observations = Observations.from_event(event=event)

        if (observations is None) and (self.observations is None):
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        elif observations is not None:
            self.add_observations(observations)

        self.write_control_file()

        cmd = ['NLLoc', self.control_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        logger.info('locating event using NonLinLoc')
        t0 = time()
        output, error = process.communicate()
        t1 = time()
        logger.info(f'done locating event in {t1 - t0:0.2f} seconds')

        t, x, y, z = read_hypocenter_file(self.output_file_path / 'last.hyp')

        scatters = read_scatter_file(self.output_file_path / 'last.scat')

        rays = None
        if calculate_rays:
            logger.info('calculating rays')
            t0_ray = time()
            rays = self.travel_times.ray_tracer(np.array([x, y, z]))
            t1_ray = time()
            logger.info(f'done calculating rays in {t1_ray - t0_ray:0.2f} '
                        f'seconds')

        uncertainty_ellipsoid = calculate_uncertainty(scatters[:, :-1])

        if delete_output_files:
            for fle in self.observation_path.glob('*'):
                fle.unlink()
            self.observation_path.rmdir()

            output_dir = self.output_file_path

            for fle in output_dir.glob('*'):
                fle.unlink()
            output_dir.rmdir()

            for fle in self.output_file_path.parent.glob('*'):
                fle.unlink()

            self.output_file_path.parent.rmdir()

        result = {'event_id': str(t),
                  'time': t,
                  'hypocenter_location': np.array([x, y, z]),
                  'scatters': scatters,
                  'rays': rays,
                  'uncertainty': uncertainty_ellipsoid}

        self.last_location = result

        return result

    def rays(self, hypocenter_location):
        return self.travel_times.ray_tracer(hypocenter_location)

    @property
    def nlloc_files(self):
        return NllocInputFiles(self.observation_file,
                               self.travel_time_grid_location /
                               self.network_code,
                               self.output_file_path / self.network_code)

    @property
    def control(self):

        if self.srces is None:
            raise ValueError('The project does not contain sensors or '
                             'inventory. Sensors (srces) or inventory '
                             'information can be added using the add_srces or'
                             'add_inventory methods.')

        if self.observations is None:
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        observations = str(self.observations)

        ctrl = ''
        ctrl += str(self.control_template['control']) + '\n'
        ctrl += str(self.control_template['transformation']) + '\n\n'

        if self.control_template['locsig'] is not None:
            ctrl += self.control_template['locsig']

        if self.control_template['loccom'] is not None:
            ctrl += self.control_template['loccom'] + '\n'

        ctrl += str(self.srces) + '\n'

        ctrl += str(self.nlloc_files) + '\n'

        ctrl += str(self.control_template['locsearch'])
        ctrl += str(self.control_template['locmeth'])
        ctrl += str(self.control_template['locgau']) + '\n'

        ctrl += str(self.control_template['locqual2err'])

        if self.p_velocity is not None:
            ctrl += str(LocGrid.init_from_grid(self.p_velocity))
        else:
            raise ValueError('Cannot initialize the LocGrid, the velocity '
                             'grids are not defined')

        return ctrl