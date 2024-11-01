#!/bin/bash

cd "$(dirname "$0")"
echo "Resetting environment (if nessesary)"
python delete_agents.py
python delete_lambda_functons.py
python delete_guard_rails.py
echo "Environment reset complete."
