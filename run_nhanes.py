import subprocess

# Commands
commands_nhanes = [
    "python",
    "main_nhanes.py",
    "--weight_models", "xgboost",
    "--gamma_list", "1", "1.5", "2",
    "--y_max", "20",
    "--decision", "0", "1",
    "--name", "nhanes",
]

# Run commands
subprocess.run(commands_nhanes)
subprocess.run(commands_nhanes + ["--skip_x", "age"])
subprocess.run(commands_nhanes + ["--skip_x", "income"])
subprocess.run(commands_nhanes + ["--skip_x", "smoking.now"])
subprocess.run(commands_nhanes + ["--skip_x", "gender"])
subprocess.run(commands_nhanes + ["--skip_x", "education"])
