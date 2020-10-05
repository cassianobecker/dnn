# Framework

The Framework is a job scheduler that lets you run experiments locally or on a CBICA cluster with a configuration file. There are two parts of the framework: the scheduler `schedule.py` and your experiment class. The general structure of the framework is
```
├──schedule.py
    ├──execution environment
    └──configuration file
└──YourExperiment.py
    └──YourExperiment.execute()
```

## Script `schedule.py`

`python schedule.py` is the entry point to the framework.

The script takes two parameters: (1) the execution environment and (2) the URL for the configuration file.

Example: `python schedule.py local args.ini`

### (1) Execution environment

| Tag         | Description |
|-------------|-------------|
| `local`     | Runs experiment as `python -c`. |
| `debug`     | Runs experiment directly. Outputs to console. |
| `cbica_cpu` | Submits experiment via `qsub`. |
| `cbica_gpu` | Submits experiment via `qsub` with the gpu. |

### (2) Configuration file

Configuration files use the [`.ini`](https://en.wikipedia.org/wiki/INI_file) format and extension. Basically, you have sections and key-value pairs.

Example:
``` ini
[Section]
key=value
```

You can have multiple values for each key. The framework will create subdirectories for those values and run separate experiments. In general, you want to have a configuration file something like:
``` ini
[EXPERIMENT]
short_name = short-name
long_name = Longer description
experiment_class_name = experiments.path.to.YourExperiment

[OUTPUTS]
base_path = ~/path/to/results

[ALGORITHM]
some_parameters_for_algorithms = one | two | three

[OTHER]
other_parameters = more

[METRICS]
metrics_ini_url = experiments/hcp/conf/metrics.ini
```

It is customary to put `.ini` files in a separate `conf` directory.

## Class `Experiment`

The `Experiment` class is located in `fwk/experiment.py` and executes your experiments.

In your class, you get (1) your `execute()` method run by `Experiment`, (2) easy access to your configuration file via `Config`, and (3) `MetricsHandler`.

### (1) `execute()`

To run an experiment,
1. create a class
2. write your experiment code in `def execute(self):`.

Example:
``` python
class Example:

def execute(self):
  print('Hello world!')
```

### (2) `Config`

You can use `Config.config['your_section']['your_key']` to access your configuration file.

Example:
``` python
from fwk.config import Config

class Example:

def execute(self):
  print(f"Here is a configuration key: {Config.config['Section']['key']}")
```

Alternatively, you can also use the convenience method `Config.get_option('your_section', 'your_key', cast_function=None, default=None)`. This method returns the value specified in the `default` parameter in case the session-key pair is not present. It also lets you specify a casting function to transform the string returned by the key into the desired data type. The last two arguments are optional, and can be omitted, if desired.

### (3) `MetricsHandler`

You can use `MetricsHandler.dispatch_event(locals(), 'event_name')` to log local variables.

The following event names are built-in: `before_setup`, `after_setup`, `before_epoch`, `after_epoch`, `before_train_batch`, `after_train_batch`, `before_test_batch`, `after_test_batch`.

Example:
``` python
from fwk.metrics import MetricsHandler

class Example:

def execute(self):
  MetricsHandler.dispatch_event(locals(), 'before_setup')
  self.setup()
  MetricsHandler.dispatch_event(locals(), 'after_setup')
  
def setup(self):
  print('Setting up...')
```
