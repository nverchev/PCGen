#  start the server to show the learning curves (see online for more options)
python visdom

# New command line
# Running best model. Run python train_eval_main_model.py AE --h for more options.
python train_eval_main_model.py AE --decoder PCGen --final --components 8 --exp PCGen8 --filtering
python visualize_reconstructions.py AE AE --decoder PCGen --final --components 8 --exp PCGen8 \
                                   --interactive_plot --viz 4 555 1762 1988 2435 3827 6348 7541 --seed 3407
