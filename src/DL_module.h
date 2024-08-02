/*
    cnn.h
    Layer creation and trainig module of CNN
*/
#ifndef DL_MODULE_H
#define DL_MODULE_H

#include<stdbool.h>
#include"cJSON.h"
#include"utils.h"

typedef enum _LayerType {
    INPUT = 10,
    DENSE,
    CONV1D,
    CONV2D,
    MAXPOOLING1D,
    MAXPOOLING2D,
    BATCHNORM,
    RELU,
    SOFTMAX,
    SIGMOID,
    DROPOUT,
    FLATTEN
};

typedef enum _LayerActivationDetail{
    INPUT_ACT = 10,
    OUTPUT_ACT,
    RELU_ACT,
    SIGMOID_ACT,
    SOFTMAX_ACT,
    ETC_ACT
};

typedef enum _LossFunction {
    CROSS_ENTROPY_ERROR = 10
};

typedef enum _Optimizer_num{
    ADAM = 10
};

typedef enum _Regularizer_num{
    L1 = 10,
    L2
};

typedef enum _Padding_Type {
    SAME = 10,
    VALID
    //CUSTOM
};



typedef struct _Conv1d Conv1d;
typedef struct _Conv2d Conv2d;
typedef struct _Dense Dense;
typedef struct _BatchNormalization BatchNormalization;
typedef struct _MaxPooling1d MaxPooling1d;
typedef struct _MaxPooling2d MaxPooling2d;
typedef struct _Relu Relu;
typedef struct _Sigmoid Sigmoid;
typedef struct _Softmax Softmax;
typedef struct _Input Input;
typedef struct _Dropout Dropout;
typedef struct _Flatten Flatten;

typedef struct _Layer Layer;

typedef struct _Optimizer Optimizer;
typedef struct _Adam Adam;


struct _Sigmoid{};

struct _Conv1d{
    //need codes
};

struct _Conv2d{
    Matrix *W;
    Matrix *b;
    int padding_h_up,padding_h_down, padding_w_left,padding_w_right;
    int stride_h, stride_w;

    int filter_size_h, filter_size_w;
    int filter_num;
    int padding_type;

    Matrix *col;
    Matrix *col_W;

    Matrix *dW;
    Matrix *db;

    Matrix *mW;
    Matrix *vW;
    Matrix *mb;
    Matrix *vb;
};

struct _Dense{
    int nodes;
    Matrix* W;  //weights
    Matrix* b;  //bias

    Matrix *x;  //inputs

    Matrix *dW;
    Matrix *db;
    int loss_function;
    double loss;

    Matrix *mW;
    Matrix *vW;
    Matrix *mb;
    Matrix *vb;
};

struct _BatchNormalization{
    Matrix *gamma;
    Matrix *beta;
    double momentum;    //default=0.9
    
    Matrix *running_mean;
    Matrix *running_var;

    Matrix *xc;
    Matrix *std;
    Matrix *dgamma;
    Matrix *dbeta;
    Matrix *xn;

    Matrix *mgamma;
    Matrix *vgamma;
    Matrix *mbeta;
    Matrix *vbeta;
};

struct _MaxPooling1d{
    //need codes
};

struct _MaxPooling2d{
    int padding_h_up,padding_h_down, padding_w_left,padding_w_right;
    int stride_h, stride_w;
    int pool_h, pool_w;
    int padding_type;

    Matrix *x;
    int *arg_max;
    int arg_max_size;
};

struct _Relu{
    bool *mask;
};

struct _Softmax{
    double loss;
    Matrix *y;
    Matrix *t;
    int nodes;
    int loss_function;
};

struct _Input{};

struct _Dropout{
    bool *mask;
    double dropout_ratio;
};

struct _Flatten{

};

struct _Layer{
    int idx;
    Layer*prev;
    Layer*next;
    
    Matrix *out;
    Matrix *dout;
    int batchsize;
    int layer_type;
    int layer_activation_detail;

    union {
        Conv1d conv1d;
        Conv2d conv2d;
        Dense dense;
        BatchNormalization batchnormalization;
        MaxPooling1d maxpooling1d;
        MaxPooling2d maxpooling2d;
        Relu relu;
        Sigmoid sigmoid;
        Softmax softmax;
        Input input;
        Dropout dropout;
        Flatten flatten;
    };
};



struct _Adam{
    double lr;
    double beta1;
    double beta2;
    int iter;
};

struct _Optimizer{
    int opt_name;

    union
    {
        Adam adam;
    };
    
};

/*
    initial setting of model
*/
void init_layer(Layer** _layer);
void create_model(Layer** model);


/*
    layer addition
*/
//void add_conv1d(Layer* model);
void add_conv2d(Layer* model, int _filters,int _filter_size_w, int _filter_size_h, int _padding_type, int _stride_w, int _stride_h);
void add_dense(Layer* model, int nodes);
//void add_dense_with_loss(Layer* model, int nodes);
void add_batchnormalization(Layer* model, double momentum);     //if momentum-> -1: 0.99 for default
//void add_maxpooling1d(Layer* model);
void add_maxpooling2d(Layer* model,int _pool_w, int _pool_h, int _padding_type, int _stride_w, int _stride_h);
void add_relu(Layer* model);
void add_sigmoid(Layer* model);
void add_softmax_with_loss(Layer* model, int class, int loss_function);
void add_flatten(Layer* model);
void add_dropout(Layer* model, double dropout_ratio);

/*
    initialization of model with batchsize
*/
void init_model(Layer* model, int *input_shape, int input_dim, bool init_weight);
void init_input(Layer* cur, int * input_shape, int input_dim);
//void init_conv1d(Layer* pre, Layer* cur, bool init_weight);
void init_conv2d(Layer* pre, Layer* cur, bool init_weight);
void init_dense(Layer* pre, Layer* cur, bool init_weight);
//void init_dense_with_loss(Layer* pre, Layer* cur, Matrix *t, bool init_weight);
void init_batchnormalization(Layer* pre, Layer* cur, bool init_weight);
//void init_maxpooling1d(Layer* pre, Layer* cur);
void init_maxpooling2d(Layer* pre, Layer* cur);
void init_relu(Layer* pre, Layer* cur);
void init_sigmoid(Layer* pre, Layer* cur);
//void init_softmax(Layer* pre, Layer* cur);
void init_softmax_with_loss(Layer* pre, Layer* cur);
void init_flatten(Layer* pre, Layer* cur);
void init_dropout(Layer* pre, Layer* cur);


/*
    forward
*/
double* forward(Layer* model, Matrix *input, Matrix *t,int batchsize, bool train_flg);
void forward_input(Layer* cur, Matrix* input);
//void forward_conv1d(Layer* pre, Layer* cur);
void forward_conv2d(Layer* pre, Layer* cur);
void forward_dense(Layer* pre, Layer* cur);
//void forward_dense_with_loss(Layer* pre, Layer* cur, Matrix *t);
void forward_batchnormalization(Layer* pre, Layer* cur, bool train_flg);
//void forward_maxpooling1d(Layer* pre, Layer* cur);
void forward_maxpooling2d(Layer* pre, Layer* cur);
void forward_relu(Layer* pre, Layer* cur);
void forward_sigmoid(Layer* pre, Layer* cur);
//void forward_softmax(Layer* pre, Layer* cur);
double* forward_softmax_with_loss(Layer* pre, Layer* cur, Matrix *t);
void forward_flatten(Layer* pre, Layer* cur);
void forward_dropout(Layer* pre, Layer* cur, bool train_flg);

/*
    backward
*/
void backward(Layer* backward_start);
//void backward_conv1d(Layer* pre, Layer* next, Layer* cur);
void backward_conv2d(Layer* pre, Layer* next, Layer* cur);
void backward_dense(Layer* pre, Layer* next, Layer* cur);
//void backward_dense_with_loss(Layer* pre, Layer* cur);
void backward_batchnormalization(Layer* pre, Layer* next, Layer* cur);
//void backward_maxpooling1d(Layer* pre, Layer* next, Layer* cur);
void backward_maxpooling2d(Layer* pre, Layer* next, Layer* cur);
void backward_relu(Layer* pre, Layer* next, Layer* cur);
void backward_sigmoid(Layer* pre, Layer* next, Layer* cur);
// void backward_softmax(Layer* pre, Layer* next, Layer* cur);
void backward_softmax_with_loss(Layer* pre, Layer* cur);
void backward_flatten(Layer* pre, Layer* next, Layer* cur);
void backward_dropout(Layer* pre, Layer* next, Layer* cur);


/*
    make empty model
*/
void make_empty_model(Layer* model, bool without_weight);
void make_empty_input(Layer* cur);
//void make_empty_conv1d(Layer* cur, bool init_weight);
void make_empty_conv2d(Layer* cur, bool without_weight);
void make_empty_dense(Layer* cur, bool without_weight);
//void make_empty_dense_with_loss(Layer* cur, bool without_weight);
void make_empty_batchnormalization(Layer* cur, bool without_weight);
//void make_empty_maxpooling1d(Layer* cur);
void make_empty_maxpooling2d(Layer* cur);
void make_empty_relu(Layer* cur);
void make_empty_sigmoid(Layer* cur);
//void make_empty_softmax(Layer* cur);
void make_empty_softmax_with_loss(Layer* cur);
void make_empty_flatten(Layer* cur);
void make_empty_dropout(Layer* cur);

/*
    regularization
*/
void regularization(Layer* model,double *loss,int regularization_type, double lambda);  //default 0.01 -> input DEFAULT (-1) makes default;
void regularization_weight(Layer* model, int regularization_type, double lambda);   //default 0.01 -> input DEFAULT (-1) makes default;

/*
    optimization
*/

void init_adam_optimizer(Optimizer** opt, double _lr, double _beta1, double _beta2);   //default: 0.001, 0.9, 0.999
void init_adam_params(Layer* model);
void adam_update(Layer *model, Adam *adam_opt);



/*
    Utils
*/
Layer* model_tail(Layer* model);
void delete_model(Layer* model);
double classification_accuracy(Layer*model, Matrix *x, Matrix*_t, double **loss);    //any double param for loss
/*return prediction true / pred (int matrix form) */
int** classification_accuracy_with_batch(Layer*model, Matrix *x, Matrix*_t, double *acc, double *loss, int batchsize,int* pred_num, bool save_prediction); //any double param for loss and acc


/*
    model training
*/
double** model_fit(Layer * model, Matrix* x_train, Matrix *y_train, Matrix *x_val, Matrix *y_val, int epochs, int batchsize, Optimizer* opt,int regularization_type,double regularization_lambda ,int verbose, bool train_shuffle, const char* model_file_path);

/*
    model save/load
*/
void save_model(Layer *model, const char* filename);
Layer* load_model(const char* filepath);

#endif
