import os
import subprocess
from string import Template

create_directories = True
submit_jobs = True
huc2_list = ['01']
#huc2_list = ['02','03','04','05','06','07','08','09','10L']#['01']
#huc2_list = ['10U','11','12','13','14','15','16','17','18']

# Note: Change the path accordingly to save the submit scripts and output messages
for huc2 in huc2_list:
    if create_directories:
        os.makedirs(f'/scratch/ros_project/submit_scripts/huc{huc2}',exist_ok=True)

        with open(os.path.join('submit_template.sh')) as submit_template:
            template = Template(submit_template.read())

        with open(os.path.join(f'/scratch/ros_project/submit_scripts/huc{huc2}', f'submit.sh'), 'w') as submit_script:
            submit_script.write(template.substitute(**{
                'HUC2_LIST': str(huc2)
                }))

    if submit_jobs:
        curdir = os.getcwd()
        os.chdir(f'/scratch/ros_project/submit_scripts/huc{huc2}')
        subprocess.run(['sbatch', f'submit.sh'])
        os.chdir(curdir)
