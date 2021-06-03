#!/usr/bin/env bash

venv="venv"$1
package_name=$2

# Delete old environment directory
rm -rf ".${venv}"
# Delete old package info directory
rm -rf "${package_name//-/_}".egg-info
echo "Creating a environmen's directorgit pushy .${venv}"

echo 'Installing virtual environment'
virtualenv --python=/usr/bin/python"$1" ".${venv}"
if [[ -d ".$venv" ]]; then
  echo "Setup the environment for the project: ${package_name}"
  source ".${venv}/bin/activate" || exit 1
  pip3 install -e . || exit 2
  echo "Installation successful"
else
  printf "Something went wrong with virtual environment installation. Please repeat next command:\n
          python manage.py init"
fi
