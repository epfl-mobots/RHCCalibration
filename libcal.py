"""
Date: 2025-10-19
Author: Cyril Monette
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from configparser import ConfigParser

HEATER_KEYS = [f"h{i:02d}" for i in range(10)]


def _robotCfgFilename(board_id: str) -> str:
    """New-style per-robot config files use an 'rhc' filename prefix (e.g. RHCConfigs/rhc31.cfg
    for board_id 'abc31') even though board_id itself keeps its historical 'abc' prefix."""
    return board_id.replace("abc", "rhc", 1) + ".cfg"


def _robotCfgPath(board_id: str) -> Path:
    return Path(__file__).parent.resolve() / "RHCConfigs" / _robotCfgFilename(board_id)


def _readRobotCfg(board_id: str, logger_func=None):
    """Reads the per-robot config file (RHCConfigs/<rhc-name>.cfg) for the given board_id.
    Returns None (and warns via logger_func, if given) if no such file exists yet."""
    cfg_path = _robotCfgPath(board_id)
    if not cfg_path.exists():
        if logger_func is not None:
            logger_func(f"No RHCConfigs file found for board_id {board_id} ({cfg_path}).", level='WRN')
        return None
    cfg = ConfigParser()
    cfg.read(cfg_path)
    return cfg


def writeCalibration(board_id: str, max_htr_powers, base_power: float):
    """
    Writes newly computed heater calibration data into the robot's RHCConfigs/<rhc-name>.cfg file,
    creating the file (and RHCConfigs/ folder) if it doesn't exist yet. Preserves any mcu_uuid /
    exclude_sensors already recorded for that robot. `max_htr_powers` is indexed by h00..h09 (e.g.
    the pd.Series returned by findMaxHtrPowers).
    """
    cfg_path = _robotCfgPath(board_id)
    cfg_path.parent.mkdir(exist_ok=True)

    existing = ConfigParser()
    if cfg_path.exists():
        existing.read(cfg_path)
    mcu_uuid = existing.get("Board", "mcu_uuid", fallback="")
    excl = existing.get("InvalidSensors", "exclude_sensors", fallback="")

    htr_lines = "\n".join(f"{h} = {max_htr_powers[h]}" for h in HEATER_KEYS)

    content = f"""# Per-robot config file for board_id {board_id}.
# Maintained by RHCCalibration's CalibrationFilesGenerator.ipynb.
# Do NOT comment on the right side of the equal signs.

[Board]
board_id = {board_id}
# The MCU_UUID is acquired through the conversion of the 3 hex values from the MCU to binary; then appending them together and converting the result to decimal.
# Use compute_MCU_uuid.py to calculate the MCU_UUID from the 3 hex values from the get_mcu_uuid() function.
mcu_uuid = {mcu_uuid}

[MaxPwmPower]
# Power (W) at PWM=950 for each heater, and the frame's base power draw (W) at PWM=0.
# Generated from InfluxDB heater-response loggings, see CalibrationFilesGenerator.ipynb.
{htr_lines}
base = {base_power}

[InvalidSensors]
# To exclude some sensors from being sampled, e.g. if they are not present or not working,
# enter their indices here separated by space. This will disable the sensors through the fw at init.
# To ignore t10 and t30, for example, one could put:
# exclude_sensors = 10 30
exclude_sensors = {excl}
"""
    with open(cfg_path, "w") as f:
        f.write(content)


def getConversionRatios(board_id:str, direction:str="PwrToPwm", verbose:bool=False)->dict:
    """
    Reads the calibration values from the calibration file corresponding to the given board_id.
    Returns a dictionary with the conversion ratios that allow to find a pwm from a pwr input
    (direction="PwrToPwm", unit pwm/W) or a pwr from a pwm input (direction="PwmToPwr", unit W/pwm)
    for every heater of the frame.
    """
    assert direction in ("PwrToPwm", "PwmToPwr"), "direction must be either 'PwrToPwm' or 'PwmToPwr'"
    cfg = _readRobotCfg(board_id)
    if cfg is None or not cfg.has_section("MaxPwmPower") or not cfg.get("MaxPwmPower", "h00", fallback=""):
        raise KeyError(f"No MaxPwmPower calibration data found for board_id {board_id} in RHCConfigs.")
    if verbose:
        print(f"[D] Calibration data for board_id {board_id}:\n{dict(cfg['MaxPwmPower'])}")
    conversion_ratios = {}
    for htr in HEATER_KEYS:
        power = cfg.getfloat("MaxPwmPower", htr)
        if direction == "PwrToPwm":
            ratio = 950 / power # Unit is pwm/W
        else:
            ratio = power / 950 # Unit is W/pwm
        conversion_ratios[htr] = round(ratio, 4)

    if verbose:
        print(f"[D] Conversion ratios for board_id {board_id}:\n{conversion_ratios}")

    return conversion_ratios

def getUUIDFromBoardID(board_id:str, logger_func = None):
    cfg = _readRobotCfg(board_id, logger_func)
    uuid_str = cfg.get("Board", "mcu_uuid", fallback="") if cfg is not None else ""
    if uuid_str:
        mcu_uuid = int(uuid_str)
        if logger_func is not None:
            logger_func(f"Target MCU UUID is: {mcu_uuid}", level='INF')
        return mcu_uuid
    if logger_func is not None:
        logger_func("The board_id in the cfg file does not match any board_id with a known MCU UUID in RHCConfigs. Config file serial_id used to connect to the ABC device.", level='WRN')
    return None

def getExcludedSensors(board_id:str, logger_func = None) -> list:
    """
    Returns the list of temperature sensor indices to exclude for the given board_id, read from
    its RHCConfigs/<rhc-name>.cfg file. Empty list if none configured, or if the robot has no
    RHCConfigs file yet.
    """
    cfg = _readRobotCfg(board_id, logger_func)
    excl_str = cfg.get("InvalidSensors", "exclude_sensors", fallback="") if cfg is not None else ""
    return [int(s) for s in excl_str.split()] if excl_str else []

def frameBasePower(abc_data:pd.DataFrame):
    """
    Returns the base power consumption (W) when PWM=0 for the given 
    """
    power = abc_data[(abc_data["_measurement"] == "pwr") & (abc_data["_field"] == "power")]
    pwm = abc_data[(abc_data["_measurement"] == "htr")& (abc_data["_field"] == "pwm")]
    power_times = power.index.unique()
    pwm_times = pwm.index.unique()
    unique_times = pd.Index(sorted(set(power_times) & set(pwm_times)))

    no_power_dt = []
    for dt in unique_times:
        _pwm = pwm.loc[dt, "_value"]
        if all(_pwm == 0):
            no_power_dt.append(dt)

    power0_df = power.loc[no_power_dt]
    # Remove the outliers (_value > 1 W)
    power0_df = power0_df[power0_df["_value"] < 1]
    assert len(power0_df) > 10, "Not enough data points with PWM=0"
    power0 = power0_df["_value"].median()

    # Plotting
    plt.figure()
    scatter_plot = sns.scatterplot(data=power0_df, x=power0_df.index, y=power0_df["_value"])
    scatter_plot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    scatter_plot.tick_params(axis='x', rotation=45)
    plt.axhline(y=power0, color='r', linestyle='--')
    scatter_plot.set_title("Base Power Consumption at PWM=0")
    scatter_plot.set_ylabel("Power (W)")
    scatter_plot.set_xlabel("Time")

    return round(power0, 3)

def findMaxHtrPowers(abc_data:pd.DataFrame, base_power:float):
    """
    Returns the max heater powers (W) for the given ABC data
    """
    power = abc_data[(abc_data["_measurement"] == "pwr") & (abc_data["_field"] == "power")]
    pwm = abc_data[(abc_data["_measurement"] == "htr") & (abc_data["_field"] == "pwm")]

    htrs = sorted(pwm["actuator_instance"].unique())
    max_powers = pd.Series()
    plt.figure()
    for htr in htrs:
        pwm_htr = pwm[pwm["actuator_instance"] == htr]
        pwm_other_htrs = pwm[pwm["actuator_instance"] != htr]
        
        power_times = power.index.unique()
        pwm_times = pwm.index.unique()
        unique_times = pd.Index(sorted(set(power_times) & set(pwm_times)))
        max_power_dt = []
        for dt in unique_times:
            _pwm = pwm_htr.loc[dt]
            _pwm_others = pwm_other_htrs.loc[dt]
            # Check if all other heaters are at PWM=0
            if (_pwm["_value"] == 950).any() and (_pwm_others["_value"] == 0).all():
                max_power_dt.append(dt)

        max_pwm_power_df = power.loc[max_power_dt]
        # Remove all values below 1.5 W or above 3.0 W (assumed to be outliers)
        max_pwm_power_df = max_pwm_power_df[(max_pwm_power_df["_value"] > 1.5) & (max_pwm_power_df["_value"] < 3.0)]
        assert len(max_pwm_power_df) >= 4, f"Not enough data points with PWM=950 for heater {htr}: found {len(max_pwm_power_df)}"
        
        # Plotting
        scatter_plot = sns.scatterplot(data=max_pwm_power_df, x=max_pwm_power_df.index, y=max_pwm_power_df["_value"])
        scatter_plot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        scatter_plot.tick_params(axis='x', rotation=45)
        scatter_plot.set_title(f"Heater {htr} Power at PWM=950")
        scatter_plot.set_ylabel("Power (W)")
        scatter_plot.set_xlabel("Time")
        max_pwm_power = max_pwm_power_df["_value"].median() - base_power
        max_powers.at[htr] = round(max_pwm_power, 3)

    return max_powers
