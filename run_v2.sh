#!/bin/bash
cd /home/naritadaiki3/nexifuse_project
exec nexifuse_env/bin/python -m nexifuse pipeline-v2 --data-only -w 2 >> v2_pipeline.log 2>&1
