# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, sys
import base64, json, requests
import calendar, time
import argparse, subprocess

_api_base_url = 'https://api.ngc.nvidia.com'

def _get_token(org, team):
    '''Use the NGC_CLI_API_KEY environment variable to generate auth token'''

    url = 'https://authn.nvidia.com/token'
    auth = '$oauthtoken:{0}'.format(os.environ.get('NGC_CLI_API_KEY'))

    headers = {
        'Authorization': 'Basic {}'.format(base64.b64encode(auth.encode('utf-8')).decode('utf-8')),
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
    }

    scope = f'group/ngc:{org}'
    if team:
        scope += f'/{team}'

    querystring = {"service": "ngc", "scope": scope}

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        request_status = response.json()['requestStatus']
        raise Exception("HTTP Error %d from '%s': [%s] %s" % (response.status_code, url, request_status['statusCode'], request_status['statusDescription']))

    return json.loads(response.text.encode('utf8'))["token"]

def _get_job_info(org, job_id, token):
 
    url = f'{_api_base_url}/v2/org/{org}/jobs/{job_id}'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        request_status = response.json()['requestStatus']
        raise Exception("HTTP Error %d from '%s': [%s] %s" % (response.status_code, url, request_status['statusCode'], request_status['statusDescription']))

    return response.json()

def _create_job(org, team, name, image, command, workspace, port, priority, order, ace, instance, runtime, token):
 
    url = f'{_api_base_url}/v2/org/{org}/team/{team}/jobs/'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    querystring = {
        "userLabels": [
            "wl___other___quantum"
        ],
        "aceName": ace,
        "aceInstance": instance,
        "dockerImageName": image,
        "jobOrder": order,
        "workspaceMounts": [
            {
                "containerMountPoint": f'/workspaces/{workspace}',
                "id": workspace,
                "mountMode": "RW"
            }
        ],
        "description": "string",
        "replicaCount": 0, # for multi-node jobs
        "publishedContainerPorts": [
            port
        ],
        "jobPriority": priority.upper(),
        "name": name,
        "command": command,
        "runPolicy": {
            "totalRuntimeSeconds": runtime,
            "preemptClass": "RUNONCE"
        },
        "resultContainerMountPoint": "/results"
    }

    response = requests.post(url, headers=headers, data=querystring)
    if response.status_code != 200:
        request_status = response.json()['requestStatus']
        raise Exception("HTTP Error %d from '%s': [%s] %s" % (response.status_code, url, request_status['statusCode'], request_status['statusDescription']))

    return response.json()

def query(command_line):

    timestamp = calendar.timegm(time.gmtime())
    parser = argparse.ArgumentParser(
        prog='python3 ngc.py',
        description='Helper script to interact with NGC.',
        epilog='For more information about NGC, see https://docs.nvidia.com/base-command-platform/user-guide/index.html')
    subparsers = parser.add_subparsers(dest='verb', help='subcommands')

    parser.add_argument("--version", action="version", version="Oct 2023")
    parser.add_argument("--org", help='The NGC organization name. Default: nvidian', default='nvidian')
    parser.add_argument("--team", help='The NGC team name. Default: gsw-prd81', default='gsw-prd81')

    job_parser = subparsers.add_parser('job').add_subparsers(dest='job', help='subcommands')
    info = job_parser.add_parser('info', help='Print information about an existing job.')
    info.add_argument('--id', type=int, required=True, help='The job id.')

    create = job_parser.add_parser('create', help='Create a new job.')
    create.add_argument('--image', type=str, required=False, default='nvidia/nightly/cuda-quantum:latest',
                        help='The image to load. Default: nvidia/nightly/cuda-quantum:latest')
    create.add_argument('--name', type=str, required=False, default=f'Job-{timestamp}',
                        help=f'The job name. Default: Job-{timestamp}')
    create.add_argument('--priority', nargs='?', default='normal', const='normal',
                        choices=['low', 'normal', 'high'],
                        help='The job priority. Default: normal')
    create.add_argument('--order', type=int, required=False, default=50,
                        help='Interger determining the job order. Jobs are ordered from 1 to 99. Default: 50')
    create.add_argument('--ace', nargs='?', default='nv-us-east-3', const='',
                        choices=['nv-us-east-3', 'no-ace'],
                        help='The ACE name. Default: nv-us-east-3')
    create.add_argument('--instance', nargs='?', default='dgxa100.80g.2.norm', const='',
                        choices=['dgxa100.80g.2.norm', 'dgxa100.80g.4.norm', 'dgxa100.80g.8.norm'],
                        help='The ace instance name. Default: dgxa100.80g.1.norm')
    create.add_argument('--runtime', type=int, required=False, default=60,
                        help='Maximum runtime in seconds that the job is in the running state. Default: 60')
    create.add_argument('--command', type=str, required=False, default='echo "No command to invoke."',
                        help='The command to execute. Default: None')
    create.add_argument('--workspace', type=str, required=False, default='cuda-quantum',
                        help='Workspace bound to the job. Default: cuda-quantum')
    create.add_argument('--port', type=int, required=False, default=8888,
                        help='Ports to open on the docker container. Default: 8888')

    args = parser.parse_args(command_line.split())
    if (not "NGC_CLI_API_KEY" in os.environ) or os.environ['NGC_CLI_API_KEY'] == '':
        print("Please set the environment variable NGC_CLI_API_KEY to a valid API key.", file=sys.stderr)
        sys.exit(1)

    token=_get_token(org=args.org, team=args.team)
    if args.verb == 'job':
        if args.job == 'info':
            job_info=_get_job_info(org=args.org, job_id=args.id, token=token)
            print(json.dumps(job_info, indent=2))

        elif args.job == 'create':
            job_info=_create_job(org=args.org, team=args.team, name=args.name, image=args.image, command=args.command, workspace=args.workspace, port=args.port, priority=args.priority, order=args.order, ace=args.ace, instance=args.instance, runtime=args.runtime, token=token)
            print(json.dumps(job_info, indent=2))

        else:
            subparser = subparsers.choices['job']
            print(subparser.format_help())
    else:
        parser.print_help()
