import os

cmd = 'python pix2pix_cancer_classification.py --input_option 2 --run_id 50 --gpu 2' \
      ' --fold 2 --use_fake True --norm_data True --num_stage 2'
os.system(cmd)
