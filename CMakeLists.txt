cmake_minimum_required (VERSION 2.6)

cmake_policy(SET CMP0012 NEW)

PROJECT(LOL_Classification)

SET(OpenCV_DIR "D:/OpenCV/opencv340/build4vc140/install")

# Requires OpenCV
FIND_PACKAGE( OpenCV REQUIRED )

ADD_EXECUTABLE( preprocess src/main_preprocess.cpp )
TARGET_LINK_LIBRARIES( preprocess ${OpenCV_LIBS} )

ADD_EXECUTABLE( svm_train src/main_svm_train.cpp src/utils.cpp )
TARGET_LINK_LIBRARIES( svm_train ${OpenCV_LIBS} )

ADD_EXECUTABLE( svm_prediction src/main_svm_pred.cpp src/utils.cpp )
TARGET_LINK_LIBRARIES( svm_prediction ${OpenCV_LIBS} )
