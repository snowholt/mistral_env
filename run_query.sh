#!/bin/bash
cd /home/lumi/beautyai
source backend/venv/bin/activate
python query_agent_configs.py
echo "Script done, exit code: $?"
ls -la reports/agent_configs_query_result.txt 2>&1 || echo "File not found"
