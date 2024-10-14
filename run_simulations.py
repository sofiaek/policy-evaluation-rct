import subprocess

# Commands
commands_all_coverage = ['python', 'main_policy.py',
                         '--seed', '145',
                         '--n_train_rct', '1000',
                         '--n_train_rwd', '2000',
                         '--n_test', '1000',
                         '--decision', '1',
                         '--gamma_list', '1', '2',
                         '--weight_models', 'true', 'logistic', 'xgboost',
                         '--n_mc', '1000'
                         ]

commands_all_policy = ['python', 'main_policy.py',
                       '--seed', '145',
                       '--n_train_rct', '1000',
                       '--n_train_rwd', '2000',
                       '--n_test', '1000',
                       '--decision', '0', '1',
                       '--gamma_list', '1', '2',
                       '--weight_models', 'xgboost',
                       '--n_mc', '1'
                       ]

commands_all_skip_x = ['python', 'main_policy.py',
                       '--seed', '0',
                       '--n_train_rct', '1000',
                       '--n_train_rwd', '2000',
                       '--n_test', '1000',
                       '--decision', '1',
                       '--gamma_list', '1', '1.5',
                       '--weight_models', 'true', 'logistic', 'xgboost',
                       '--n_mc', '1']

commands_first = ['python', 'main_policy.py',
                  '--name', 'popAfirst',
                  '--gamma_list', '1', '1.5', '2',
                  '--seed', '145',
                  '--weight_models', 'true', 'xgboost',
                  '--n_train_rct', '1000',
                  '--n_test', '1000',
                  '--n_train_rwd', '2000',
                  '--mu_x_s1', '0.5', '0.5',
                  '--sigma_x_s1', '1.0', '1.0',
                  '--mu_u_s1', '0.5',
                  '--sigma_u_s1', '1.0',
                  '--n_mc', '1000']

commands_popA = ['--name', 'popA',
                   '--mu_x_s1', '0.5', '0.5',
                   '--sigma_x_s1', '1', '1',
                   '--mu_u_s1', '0.5',
                   '--sigma_u_s1', '1']

commands_popB = ['--name', 'popB',
                   '--mu_x_s1', '0.0', '0.5',
                   '--sigma_x_s1', '1.25', '1.5',
                   '--mu_u_s1', '0.0',
                   '--sigma_u_s1', '1.25']

commands_popC = ['--name', 'popC',
                   '--mu_x_s1', '0.0', '0.0',
                   '--sigma_x_s1', '1.5', '1.5',
                   '--mu_u_s1', '0.0',
                   '--sigma_u_s1', '1.5']

commands_popD = ['--name', 'popD',
                   '--mu_x_s1', '0.25', '0.25',
                   '--sigma_x_s1', '1', '0.25',
                   '--mu_u_s1', '0.25',
                   '--sigma_u_s1', '0.5']

# Run commands

# Coverage
subprocess.run(commands_first)
subprocess.run(commands_all_coverage + commands_popA)
subprocess.run(commands_all_coverage + commands_popB)
subprocess.run(commands_all_coverage + commands_popC)
subprocess.run(commands_all_coverage + commands_popD)

# Compare policy
subprocess.run(commands_all_policy + commands_popA)
subprocess.run(commands_all_policy + commands_popB)
subprocess.run(commands_all_policy + commands_popC)
subprocess.run(commands_all_policy + commands_popD)

# Remove covariate
subprocess.run(commands_all_skip_x + commands_popA)
subprocess.run(commands_all_skip_x + commands_popA + ['--skip_x', 'X_0'])
subprocess.run(commands_all_skip_x + commands_popA + ['--skip_x', 'X_1'])

subprocess.run(commands_all_skip_x + commands_popB)
subprocess.run(commands_all_skip_x + commands_popB + ['--skip_x', 'X_0'])
subprocess.run(commands_all_skip_x + commands_popB + ['--skip_x', 'X_1'])

subprocess.run(commands_all_skip_x + commands_popC)
subprocess.run(commands_all_skip_x + commands_popC + ['--skip_x', 'X_0'])
subprocess.run(commands_all_skip_x + commands_popC + ['--skip_x', 'X_1'])

subprocess.run(commands_all_skip_x + commands_popD)
subprocess.run(commands_all_skip_x + commands_popD + ['--skip_x', 'X_0'])
subprocess.run(commands_all_skip_x + commands_popD + ['--skip_x', 'X_1'])