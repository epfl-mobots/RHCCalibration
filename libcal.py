"""
Date: 2025-10-19
Author: Cyril Monette
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def getConversionRatios(board_id:str, direction:str="PwrToPwm", verbose:bool=False)->dict:
    """
    Reads the calibration values from the calibration file corresponding to the given board_id.
    Returns a dictionary with the conversion ratios that allow to find a pwm from a pwr input
    (direction="PwrToPwm", unit pwm/W) or a pwr from a pwm input (direction="PwmToPwr", unit W/pwm)
    for every heater of the frame.
    """
    assert direction in ("PwrToPwm", "PwmToPwr"), "direction must be either 'PwrToPwm' or 'PwmToPwr'"
    # Open the calibration file
    current_wd = Path(__file__).parent.resolve()
    calib_file = current_wd / f"max_pwm_powers.csv"
    calib_df = pd.read_csv(calib_file)
    # Set board_id col as index
    calib_df.set_index("board_id", inplace=True)
    board_calib_df = calib_df.loc[board_id, :].to_frame().T
    if verbose:
        print(f"[D] Calibration data for board_id {board_id}:\n{board_calib_df}")
    conversion_ratios = {}
    for htr in board_calib_df.columns:
        if htr == "base":
            continue
        if direction == "PwrToPwm":
            ratio = 950 / board_calib_df.loc[board_id, htr] # Unit is pwm/W
        else:
            ratio = board_calib_df.loc[board_id, htr] / 950 # Unit is W/pwm
        ratio_rounded = round(ratio, 4)
        conversion_ratios[htr] = ratio_rounded

    if verbose:
        print(f"[D] Conversion ratios for board_id {board_id}:\n{conversion_ratios}")

    return conversion_ratios

def getUUIDFromBoardID(board_id:str, logger_func = None):
        # Open the mcu_uuid-abc_ids.csv file
        # get current working directory path
        current_wd = Path(__file__).parent.resolve()
        with open(current_wd / "mcu_uuid-abc_ids.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == "#":
                    continue    # Skip comments
                if line.split(",")[0].strip() == board_id:
                    mcu_uuid = int(line.split(",")[1].strip())
                    if logger_func is not None:
                        logger_func(f"Target MCU UUID is: {mcu_uuid}",level='INF')
                    return mcu_uuid
        if logger_func is not None:
            logger_func("The board_id in the cfg file does not match any board_id in the mcu_uuid-abc_ids.csv file. Config file serial_id used to connect to the ABC device.", level='WRN')
        return None

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
