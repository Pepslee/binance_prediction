#!/usr/bin/env bash

venv="venv$1"
tmp_dir="/tmp/gts"
if [[ -d ".$venv" ]]; then
  mkdir -p ${tmp_dir}
  source ".${venv}/bin/activate"
  python -m unittest discover -s tests
  rm -r ${tmp_dir}
else
  printf "Something went wrong with virtual environment activation. Please make sure you've run next command: \n
          python manage.py init"
fi
