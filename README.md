# HazardousPredetection
#Description of model is at the paper, "Predicting Potentially Hazardous Chemical Reactions Using Explainable Neural Network"

#REQUIREMENTS
python : 3.7
rdkit : 2019.03.4
torch : 1.2.0 
sklearn : 0.23.2

#GUIDE FOR USE
1. Reaction data is from USPTO data and reaxys data
2. Cleaned data is located at ./data/toxin/ and ./data/explosive/
3. Reaction data which converted into fingerprints are located at ./data/fp/toxin/[nameoftoxin]/ and ./data/fp/explosive/[nameofexplosive]/
4. Or you can use ./data/fp/convert_to_fingerprint.py to convert reaction data into fingerprint data
5. You can train the prediction model using ./train.py
6. You can test and check result of the prediction using ./test.py

#convert_to_fingerprint.py
1. The location of positive and negative data, and output directory should be set
2. Radius us the maximum size of fingerprint substructure from center atom, and frequent regulate is minimum frequency that required for substructure to be included in library
3. Output is test, validation, training data with reactions, library describes substructures to fingerprints, test, validation, training data which converted into fingerprints

#train.py
1. The location of library, training data, validation data and output directory should be set
2. radius and frequent regulate should be same with fingerpritn converting option
3. maxlen is maximum number of molecules in reaction, and decay is option used to change gradient descent optimization
4. Output is trained path as .pth file

#test.py
1. The location of library, test data, trained path(.pth file) and output directory should be set
2. Other options should be same with option used to train model
3. Ouput is final test accuracy of model for positive and negative test set

#predict.py
1. The location of library, test reactions, trained path and output directory should be set
2. Other options should be same with option used to train model
3. Ouput is predictions
