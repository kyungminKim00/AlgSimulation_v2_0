docker run --gpus all -it --rm -p 8891:8888 -p 6006:6006 \
-u shinhanai.kyungmin \
--name tensorflow_2_4 \
-v /home/kmkim/python_projects/LearnKit/AlgSimulation_v2_0:/AlgSimulation_v2_0 \
-v /home/kmkim/app_data/AlgSimulation/rawdata:/AlgSimulation_v2_0/datasets/rawdata \
-v /home/kmkim/app_data/AlgSimulation/save:/AlgSimulation_v2_0/save \
--workdir /AlgSimulation_v2_0 \
cuda_11_0/tensorflow:tensorflow-gpu_2-4-0 bash
