# flairsyn dockerfile at 65f458c6 (new cuda 12)
FROM srassmann/dif:cu12

#RUN #conda install cuda -c nvidia
RUN pip install ninja h5py visdom dominate scikit-image scipy
