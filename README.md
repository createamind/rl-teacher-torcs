# rl-teacher-torcs
rl-teacher-torcs
# install dependency
pip install -e .
pip install -e human-feedback-api
pip install -e agents/parallel-trpo[tf]
pip install -e agents/pposgd-mpi[tf]

# start web server
python start_web.py

# tensorboard
tensorboard --logdir ~/tb/rl-teacher/
# start rl-teacher
python rl_teacher/teach.py -p human --pretrain_labels 175 -e Reacher-v1 -n human-175

