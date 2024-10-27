#include <kdd99/catboost_classifier.h>

#include <iostream>

#include <sstream>

using kdd99::CatboostClassifier;


CatboostClassifier::CatboostClassifier(const std::string& modepath)
    : model_{ModelCalcerCreate(), ModelCalcerDelete} {
    // model_ = ModelCalcerCreate();
    if (!LoadFullModelFromFile(model_.get(), modepath.c_str())) {
        std::stringstream ss;
        ss << "LoadFullModelFromFile error message:" << GetErrorString();
        throw std::runtime_error{ss.str()};
    }
    // was APT_PROBABILITY
    if (!SetPredictionType(model_.get(), APT_CLASS)) {
        std::stringstream ss;
        ss << "LoadFullModelFromFile error message:" << GetErrorString();
        throw std::runtime_error{ss.str()};        
    }
}

float CatboostClassifier::predict_proba(const features_t& features) const {
    double result[1];
    if (!CalcModelPredictionSingle(model_.get(), features.data(), features.size(), nullptr, 0, result, 1)) {
        std::stringstream ss;
        ss << "CalcModelPredictionSingle error message:" << GetErrorString();
        throw std::runtime_error{ss.str()};
    }
    return result[0];
}    