import os
import subprocess
cmd_sql='python human-feedback-api/manage.py migrate'.split(' ')
cmd_static='python human-feedback-api/manage.py collectstatic'.split(' ')
sql_file='human-feedback-api/db.sqlite3'
static_file='human-feedback-api/human_feedback_site/staticfiles'
if not os.path.exists(sql_file):
    subprocess.Popen(cmd_sql)
if not os.path.exists(static_file):
    subprocess.Popen(cmd_static)

cmd1='python human-feedback-api/manage.py runserver 0.0.0.0:8000'.split(' ')
cmd2='python human-feedback-api/video_server/run_server.py'.split(' ')

subprocess.Popen(cmd1)
subprocess.Popen(cmd2)