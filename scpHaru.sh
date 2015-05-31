scp *.c* mwang5@haru.caltech.edu:/home/mwang5/project/
scp Makefile mwang5@haru.caltech.edu:/home/mwang5/project/
scp *.sh mwang5@haru.caltech.edu:/home/mwang5/project/

ssh mwang5@haru.caltech.edu "cd project; sh -x run_project.sh"
# ssh mwang5@haru.caltech.edu