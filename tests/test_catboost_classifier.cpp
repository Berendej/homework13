#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <kdd99/catboost_classifier.h>
#include <helpers.h>

using kdd99::CatboostClassifier;
using std::clog;

void usage()
{
    std::cout << "usage fashio_mnist model_path test_data_path" << std::endl;
}

int main(int argc, char* argv[] )
{
    if ( argc < 3 )
    {
        usage();
        return 0;
    }
    std::string model_path{argv[1]};
    std::string test_data_path{argv[2]};
    auto predictor = CatboostClassifier{model_path};
    auto features = CatboostClassifier::features_t{};
    double y_pred_expected = 0.0;
    std::ifstream test_data{test_data_path};
    if( !test_data.is_open())
    {
        std::cerr << "can't open test data\n";
        return 0;
    }
    int total = 0;
    int succ = 0;
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        total++;
        auto y_pred = predictor.predict_proba(features);
#if TOO_VERBOSE
        std::cout << "y_pred " << y_pred << " y_pred_expected " 
                  << y_pred_expected << std::endl;
#endif
        float dif = abs(y_pred_expected - y_pred);
        if (dif < 0.5) // 1e-5)
        {
            succ++;
        }
    }
    float result = (float)succ / (float)total;
    std::cout << result << std::endl;
    return 0;
}
