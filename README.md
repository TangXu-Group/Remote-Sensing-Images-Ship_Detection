# Remote-Sensing-Images-Ship_Detection
## The paper can be downloaded from https://ieeexplore.ieee.org/abstract/document/9477497
## Paper: AR^2Det: An Accurate and Real-Time Rotational One-Stage Ship Detector in Remote Sensing Images.
## Please download "AR2DET.zip" for reproduction.

1. Fill the folder  HRSC2016 in the following format:

HRSC2016/

	FullDataSet/

		Allmages/
		
		Annotations/
		
	ImageSets/ 

2. The version of the packages:
	
		pytorch: 1.5.0
		python: 3.7.7
		cython: 0.29.21
		CUDA:11.0
		Numpy: 1.18.5

3. Use [box_utils/setup.py] to compile box_utils.

4. Run label_make.py or label_make.ipynb to generate labels of data.

5. Run V1_train_0.5_34_region4.ipynb to train model.

		(1) Before training, you can set the hyper-parameters, including gpu_id, resnet_type, threshold t, 
			and region_side_length. The meaning of last two parameters please review our paper.
		(2) The change of training loss and the accuracy of testing are shown in the training stage.
		(3) For convenience, the experimental log is recorded in the folder Experiments_log_xx, 
			and the best model is saved in the folder checkpoints_xx.

6. After the model is trained, you can utilize the performance_test.ipynb to generate FPS, visual results, and MAP.
		The visual results are saved in the folder predicted_results.
