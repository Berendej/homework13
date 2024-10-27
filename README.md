homework13 
Inference from trained CatBoost model.


Usage:

docker run --rm -ti -v $(pwd):/usr/src/app sdukshis/cppml
mkdir build
cd build
cmake ..
make
./bin/fashio_mnist  ../data/model.cbm ../data/test_data_catboost.txt
