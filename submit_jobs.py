#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:19:46 2018

@author: omarschall
"""

import subprocess
import os

def clear_results(job_file, data_path='/Users/omarschall/cluster_results/online-meta/'):

    job_name = job_file.split('/')[-1].split('.')[0]
    data_dir = os.path.join(data_path, job_name)

    subprocess.run(['rm', data_dir+'/*_*'])

def retrieve_results(job_file, scratch_path='/scratch/oem214/online-meta/',
               username='oem214', domain='prince.hpc.nyu.edu'):

    job_name = job_file.split('/')[-1].split('.')[0]
    data_path = '/Users/omarschall/cluster_results/online-meta/'
    data_dir = os.path.join(data_path, job_name)

    source_path = username+'@'+domain+':'+scratch_path+'library/'+job_name+'/'

    subprocess.run(['rsync', '-aav', source_path, data_dir])

def submit_job(job_file, n_array, scratch_path='/scratch/oem214/online-meta/',
               username='oem214', domain='prince.hpc.nyu.edu'):

    job_name = job_file.split('/')[-1]#
    data_path = '/Users/omarschall/cluster_results/online-meta/'
    data_dir = os.path.join(data_path, job_name.split('.')[0])

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        code_dir = os.path.join(data_dir, 'code')
        os.mkdir(code_dir)

    code_dir = os.path.join(data_dir, 'code')

    #copy main script to results dir
    subprocess.run(['rsync',
                    '-aav',
                    '--exclude', '.git',
                    '/Users/omarschall/online-meta/',
                    code_dir])

    subprocess.run(['rsync',
                    '-aav',
                    '--exclude', '.git',
                    '/Users/omarschall/online-meta/',
                    username+'@'+domain+':'+scratch_path])

    subprocess.run(['scp', job_file, username+'@'+domain+':/home/oem214/'])

    subprocess.run(['ssh', username+'@'+domain,
                    'sbatch', '--array=1-'+str(n_array), job_name])

def write_job_file(job_name,
                   sbatch_path='/Users/omarschall/online-meta/job_scripts/',
                   scratch_path='/scratch/oem214/online-meta/',
                   nodes=1, ppn=1, mem=16, n_hours=24):
    """
    Create a job file.
    Parameters
    ----------
    job_name : str
              Name of the job.
    sbatch_path : str
              Directory to store SBATCH file in.
    scratch_path : str
                  Directory to store output files in.
    nodes : int, optional
            Number of compute nodes.
    ppn : int, optional
          Number of cores per node.
    gpus : int, optional
           Number of GPU cores.
    mem : int, optional
          Amount, in GB, of memory.s
    n_hours : int, optional
            Running time, in hours.
    Returns
    -------
    jobfile : str
              Path to the job file.
    """

    job_file = os.path.join(sbatch_path, job_name + '.s')
    log_name = os.path.join('log', job_name)

    with open(job_file, 'w') as f:
        f.write(
            '#! /bin/bash\n'
            + '\n'
            + '#SBATCH --nodes={}\n'.format(nodes)
            + '#SBATCH --ntasks-per-node=1\n'
            + '#SBATCH --cpus-per-task={}\n'.format(ppn)
            + '#SBATCH --mem={}GB\n'.format(mem)
            + '#SBATCH --time={}:00:00\n'.format(n_hours)
            + '#SBATCH --job-name={}\n'.format(job_name[0:16])
            + '#SBATCH --output={}log/{}.o\n'.format(scratch_path, job_name[0:16])
            + '\n'
            + 'module purge\n'
            + 'SAVEPATH={}library/{}\n'.format(scratch_path, job_name)
            + 'export SAVEPATH\n'
            + 'module load python3/intel/3.6.3\n'
            + 'cd {}\n'.format(scratch_path)
            + 'pwd > {}.log\n'.format(log_name)
            + 'date >> {}.log\n'.format(log_name)
            + 'which python >> {}.log\n'.format(log_name)
            + 'python main.py\n'
            )

    return job_file