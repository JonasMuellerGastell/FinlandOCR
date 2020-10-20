# Finland Archival Data OCR
Project to digitize archival data from scanned document files. The core model takes three steps:
1) a binary classifier decides whether an image is indeed to be digitized or is a filler page in the folders
2) a heuristics based algorithm overlays a grid and thus extracts table cells
3) a set of stacked convolutional neural network layers followed by recurrent neural network layers  trained with CTC-loss decodes individual cell entries.

I train the model on a short extract of the 1940 Finnish agricultural census form and supplement the small sample base with synthetic data based off MNIST digits and employ a variety of data augmentation steps.

## Project Architecture
	|_DataPreprocessing
		|_CellExtract.py: the algorithm that turns a scanned image into a collation of individual cell images
		|_ ProcessCells.py: preprocess cell images to enhance decoding quality		|_ CreateDataSet.py and CreateKerasReadyData.py: turn cell images into data files suitable for training
		|_ ...
	|_ NeuralNet
		|_ utils
			|_ ... helper files used in training
		|_ FirstStage_BinaryClassifier.py: the binary classifier neural net that takes a scan and decides whether or not this scan is a relevant census page or an empty sheet/ empty table
		|_ Model_Basic.py: small CNN-RNN-CTC model, can be trained with or without data augmentation
		|_Model_SyntheticPlusRealData.py: train CNN-RNN-CTC with additional synthetic data in training set
		|_Model_TransferLearningFromSynth.py: train CNN-RNN-CTC first on synthetic data and then employ transfer learning to real data
	|_ SyntheticData
		|_... utilities to create synthetic training data

## Data Storage
At present, for anonymity/licence reasons, all training data is held offline. Future versions will include trained models and, if feasible, at least extracts of the training data.

