import os
import subprocess
from string import Template

create_directories = True
submit_jobs = True
version = 'nwm30'

# Note: Change the path accordingly to save the submit scripts and output messages
for year in range(2016,2023):
    if create_directories:
        os.makedirs(f'/scratch/ros_project/teehr/submit_scripts/{year}',exist_ok=True)

        with open(os.path.join('submit_template.sh')) as submit_template:
            template = Template(submit_template.read())

        with open(os.path.join(f'/scratch/ros_project/teehr/submit_scripts/{year}', f'submit.sh'), 'w') as submit_script:
            submit_script.write(template.substitute(**{
                'YEAR': str(year),
                #'MONTH': str(month),
                #'MONTH2': f'{month:02}',
                'VERSION': str(version)
                }))

    if submit_jobs:
        curdir = os.getcwd()
        os.chdir(f'/scratch/ros_project/teehr/submit_scripts/{year}')
        subprocess.run(['sbatch', f'submit.sh'])
        os.chdir(curdir)
