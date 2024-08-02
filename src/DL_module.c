#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "DL_module.h"
#include "utils.h"
#include "cJSON.h"


void init_layer(Layer** _layer){
    Layer * layer = NULL;

    layer = (Layer*)malloc(sizeof(Layer));
    *_layer = layer;
    layer->prev = NULL;
    layer->next = NULL;
    layer->out = NULL;
    layer->dout = NULL;
}

void create_model(Layer** _model){
    /*
        exception code of parameter is needed
    */
    Layer* model = NULL;

    init_layer(&model);
    *_model = model;
    model->layer_type = INPUT;
    model->layer_activation_detail = INPUT_ACT;
}

Layer* model_tail(Layer* model){
    Layer* last_layer = model;

    /*
        exception code of parameter is needed
    */
    while(last_layer->next){
        last_layer = last_layer->next;
    }

    // printf("%d\n", last_layer->layer_type);
    return last_layer;
}

void add_conv2d(Layer* model, int _filters,int _filter_size_w, int _filter_size_h, int _padding_type, int _stride_w, int _stride_h){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */
    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = CONV2D;
    cur_layer->layer_activation_detail = OUTPUT_ACT;

    cur_layer->conv2d.filter_size_w = _filter_size_w;
    cur_layer->conv2d.filter_size_h = _filter_size_h;
    cur_layer->conv2d.filter_num = _filters;
    cur_layer->conv2d.stride_w = _stride_w;
    cur_layer->conv2d.stride_h = _stride_h;
    if (_padding_type == DEFAULT)
        cur_layer->conv2d.padding_type = SAME;
    else
        cur_layer->conv2d.padding_type = _padding_type;

    cur_layer->conv2d.W = NULL;
    cur_layer->conv2d.b = NULL;
    cur_layer->conv2d.col = NULL;
    cur_layer->conv2d.col_W = NULL;

    cur_layer->conv2d.dW = NULL;
    cur_layer->conv2d.db = NULL;

    cur_layer->conv2d.mW = NULL;
    cur_layer->conv2d.vW = NULL;
    cur_layer->conv2d.mb = NULL;
    cur_layer->conv2d.vb = NULL;
}

void add_dense(Layer* model, int nodes){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = DENSE;
    cur_layer->layer_activation_detail = OUTPUT_ACT;

    cur_layer->dense.nodes = nodes;
    cur_layer->dense.W = NULL;
    cur_layer->dense.b = NULL;

    cur_layer->dense.x = NULL;

    cur_layer->dense.dW = NULL;
    cur_layer->dense.db = NULL;
    cur_layer->dense.loss_function = NONE;
    cur_layer->dense.loss = 0.0;

    cur_layer->dense.mW = NULL;
    cur_layer->dense.vW = NULL;
    cur_layer->dense.mb = NULL;
    cur_layer->dense.vb = NULL;
}

void add_batchnormalization(Layer* model, double momentum){    //if momentum-> DEFAULT: 0.9 for default
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = BATCHNORM;
    // model_tail(model);
    cur_layer->layer_activation_detail = ETC_ACT;

    cur_layer->batchnormalization.gamma = NULL;
    cur_layer->batchnormalization.beta = NULL;
    if(momentum == DEFAULT)
        cur_layer->batchnormalization.momentum = 0.99;
    else
        cur_layer->batchnormalization.momentum = momentum;

    cur_layer->batchnormalization.running_mean = NULL;
    cur_layer->batchnormalization.running_var = NULL;

    cur_layer->batchnormalization.xc = NULL;
    cur_layer->batchnormalization.std = NULL;
    cur_layer->batchnormalization.dgamma = NULL;
    cur_layer->batchnormalization.dbeta = NULL;
    cur_layer->batchnormalization.xn = NULL;

    cur_layer->batchnormalization.mgamma = NULL;
    cur_layer->batchnormalization.vgamma = NULL;
    cur_layer->batchnormalization.mbeta = NULL;
    cur_layer->batchnormalization.vbeta = NULL;
    // model_tail(model);
}
//void add_maxpooling1d(Layer* model);

void add_maxpooling2d(Layer* model,int _pool_w, int _pool_h, int _padding_type, int _stride_w, int _stride_h){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = MAXPOOLING2D;
    cur_layer->layer_activation_detail = ETC_ACT;

    if (_padding_type == DEFAULT)
        cur_layer->maxpooling2d.padding_type = VALID;
    else
        cur_layer->maxpooling2d.padding_type = _padding_type;
    cur_layer->maxpooling2d.pool_w = _pool_w;
    cur_layer->maxpooling2d.pool_h = _pool_h;
    cur_layer->maxpooling2d.stride_w = _stride_w;
    cur_layer->maxpooling2d.stride_h = _stride_h;
    cur_layer->maxpooling2d.x = NULL;
    cur_layer->maxpooling2d.arg_max = NULL;
}

void add_relu(Layer* model){
    Layer* cur_layer = NULL;
    
    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    // printf("relu_started\n");
    cur_layer->prev = model_tail(model);
    // printf("relu_started\n");
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = RELU;
    cur_layer->layer_activation_detail = ETC_ACT;
    
    cur_layer->relu.mask = NULL;
    while(cur_layer){
        // printf("%d\n", cur_layer->layer_activation_detail );
        if(cur_layer->layer_activation_detail == OUTPUT_ACT){
            // printf("relu add: func\n");
            cur_layer->layer_activation_detail = RELU_ACT;
            break;
        }
        cur_layer = cur_layer->prev;
    }
}

void add_sigmoid(Layer* model){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = SIGMOID;
    cur_layer->layer_activation_detail = ETC_ACT;
    while(cur_layer){
        if(cur_layer->layer_activation_detail == OUTPUT_ACT){
            cur_layer->layer_activation_detail = SIGMOID_ACT;
            break;
        }
        cur_layer = cur_layer->prev;
    }
}

void add_softmax_with_loss(Layer* model, int class, int loss_function){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = SOFTMAX;
    cur_layer->layer_activation_detail = ETC_ACT;


    cur_layer->softmax.nodes = class;
    cur_layer->softmax.loss_function = loss_function;

    cur_layer->softmax.y = NULL;
    cur_layer->softmax.t = NULL;

    while(cur_layer){
        if(cur_layer->layer_activation_detail == OUTPUT_ACT){
            cur_layer->layer_activation_detail = SOFTMAX_ACT;
            break;
        }
        cur_layer = cur_layer->prev;
    }
}

void add_flatten(Layer* model){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = FLATTEN;
    cur_layer->layer_activation_detail = ETC_ACT;
}

void add_dropout(Layer* model, double dropout_ratio){
    Layer* cur_layer = NULL;

    /*
        exception code of parameter is needed
    */

    init_layer(&cur_layer);
    cur_layer->prev = model_tail(model);
    cur_layer->prev->next = cur_layer;
    cur_layer->layer_type = DROPOUT;
    cur_layer->layer_activation_detail = ETC_ACT;

    cur_layer->dropout.mask = NULL;
    cur_layer->dropout.dropout_ratio = dropout_ratio;
}


void init_model(Layer* model, int *input_shape, int input_dim, bool init_weight){
    Layer* cur_layer = model;
    Layer* input_layer = model;
    Layer* prev_layer = model->prev;
    // printf("initializing model\n");
    
    while(cur_layer){
        // printf("initlog\n");
        cur_layer->batchsize = input_shape[0];
        
        switch (cur_layer->layer_type)
        {
        case INPUT:
            // printf("initlog: input\n");
            init_input(cur_layer, input_shape, input_dim);
            break;
        case DENSE:
            // printf("initlog: dense\n");
            if(cur_layer->dense.loss_function == NONE)
                init_dense(prev_layer, cur_layer, init_weight);
            else{
                printf("not a constructed layer\n");
                exit(2);
            }
                //forward_dense_with_loss(prev_layer, cur_layer, init_weight);
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            //forward_conv1d(prev_layer, cur_layer, init_weight);
            break;
        case CONV2D:
            // printf("initlog: conv2d\n");
            init_conv2d(prev_layer, cur_layer, init_weight);
            break;
        case MAXPOOLING1D:
            printf("not a constructed layer\n");
            exit(2);
            // forward_maxpooling1d(prev_layer, cur_layer);
            break;
        case MAXPOOLING2D:
            // printf("initlog: max2d\n");
            init_maxpooling2d(prev_layer, cur_layer);
            break;
        case BATCHNORM:
            // printf("initlog: batchnorm\n");
            init_batchnormalization(prev_layer, cur_layer, init_weight);
            break;
        case RELU:
            // printf("initlog: relu\n");
            init_relu(prev_layer, cur_layer);
            break;
        case SOFTMAX:
            // printf("initlog: softmax\n");
            if((cur_layer->softmax.loss_function) != NONE)
                init_softmax_with_loss(prev_layer, cur_layer);
            else{
                printf("not a constructed layer\n");
                exit(2);
            }
            //     forward_softmax(prev_layer, cur_layer);
            break;
        case SIGMOID:
            // printf("initlog: sig\n");
            init_sigmoid(prev_layer, cur_layer);
            break;
        case FLATTEN:
            // printf("initlog: flatten\n");
            init_flatten(prev_layer, cur_layer);
            break;
        case DROPOUT:
            // printf("initlog: dropout\n");
            init_dropout(prev_layer,cur_layer);
            break;
        default:
            printf("wrong layer type\n");
            exit(2);
            break;
        }
        prev_layer = cur_layer;
        cur_layer = cur_layer->next;
    }
    delete_matrix(input_layer->out);
    input_layer->out = NULL;
}

void init_input(Layer* cur, int * input_shape, int input_dim){
    
    if(cur->out != NULL && cur->out->matrix == NULL){
        delete_matrix(cur->out);
        cur->out = NULL;
    }
    cur->out = NULL;
    init_by_shape_arr_matrix(&(cur->out), input_shape, input_dim,SHAPE_ONLY, 0); 
}

//void init_conv1d(Layer* pre, Layer* cur, bool init_weight){}
void init_conv2d(Layer* pre, Layer* cur, bool init_weight){
    int input_size = (cur->conv2d.filter_size_h)*(cur->conv2d.filter_size_w)*(pre->out->shape[1]);
    int out_h, out_w;
    int pad_h, pad_w;
    int pre_h = pre->out->shape[2];
    int pre_w = pre->out->shape[3];

    delete_matrix(cur->conv2d.col);
    cur->conv2d.col = NULL;
    delete_matrix(cur->conv2d.col_W);
    cur->conv2d.col_W = NULL;
    delete_matrix(cur->conv2d.db);
    cur->conv2d.db = NULL;
    delete_matrix(cur->conv2d.dW);
    cur->conv2d.dW = NULL;

    if(init_weight){
        if(cur->layer_activation_detail == RELU_ACT || cur->layer_activation_detail == OUTPUT_ACT){
            init_matrix(&(cur->conv2d.W),4,HE,input_size,cur->conv2d.filter_num,pre->out->shape[1],cur->conv2d.filter_size_h,cur->conv2d.filter_size_w);
        }
        else if(cur->layer_activation_detail == SIGMOID_ACT || cur->layer_activation_detail == SOFTMAX_ACT)
            init_matrix(&(cur->conv2d.W),4,XAVIER,input_size,cur->conv2d.filter_num,pre->out->shape[1],cur->conv2d.filter_size_h,cur->conv2d.filter_size_w);
        init_matrix(&(cur->conv2d.b),1, ZEROS, 0 , cur->conv2d.filter_num);
        delete_matrix(cur->conv2d.mb);
        cur->conv2d.mb = NULL;
        delete_matrix(cur->conv2d.vb);
        cur->conv2d.vb = NULL;
        delete_matrix(cur->conv2d.mW);
        cur->conv2d.mW = NULL;
        delete_matrix(cur->conv2d.vW);
        cur->conv2d.vW = NULL;
    }
    // printf("conv2d segmentation\n");
    if(cur->conv2d.padding_type == SAME){
        pad_h = (pre_h -1)*(cur->conv2d.stride_h) + (cur->conv2d.filter_size_h) - pre_h;
        pad_w = (pre_w -1)*(cur->conv2d.stride_w) + (cur->conv2d.filter_size_w) - pre_w;
        if(pad_h%2){  //odd
            cur->conv2d.padding_h_up = pad_h/2 + 1;
            cur->conv2d.padding_h_down = pad_h/2;
        }
        else{
            cur->conv2d.padding_h_up = pad_h/2;
            cur->conv2d.padding_h_down = pad_h/2;
        }
        if(pad_w%2){  //odd
            cur->conv2d.padding_w_left = pad_w/2 + 1;
            cur->conv2d.padding_w_right = pad_w/2;
        }
        else{
            cur->conv2d.padding_w_left = pad_w/2;
            cur->conv2d.padding_w_right = pad_w/2;
        }
    }
    else{   //VALID
        cur->conv2d.padding_h_down = 0;
        cur->conv2d.padding_h_up = 0;
        cur->conv2d.padding_w_left = 0;
        cur->conv2d.padding_w_right = 0;
    }

    out_h = (pre_h+(cur->conv2d.padding_h_up) + (cur->conv2d.padding_h_down) - (cur->conv2d.filter_size_h))/(cur->conv2d.stride_h) + 1;
    out_w = (pre_w+(cur->conv2d.padding_w_left) + (cur->conv2d.padding_w_right) - (cur->conv2d.filter_size_w))/(cur->conv2d.stride_w) + 1;


    init_matrix(&(cur->out), 4, NONE, 0, cur->batchsize, cur->conv2d.filter_num, out_h, out_w);
    // printf("%lf\n", cur->out->matrix[cur->out->size -1]);
}

void init_dense(Layer* pre, Layer* cur, bool init_weight){
    int input_size = (pre->out->size)/(pre->out->shape[0]);

    delete_matrix(cur->dense.db);
    cur->dense.db = NULL;
    delete_matrix(cur->dense.dW);
    cur->dense.dW = NULL;
    delete_matrix(cur->dense.x);
    cur->dense.x = NULL;
    if(init_weight){
        delete_matrix(cur->dense.mb);
        cur->dense.mb = NULL;
        delete_matrix(cur->dense.vb);
        cur->dense.vb = NULL;
        delete_matrix(cur->dense.mW);
        cur->dense.mW = NULL;
        delete_matrix(cur->dense.vW);
        cur->dense.vW = NULL;
        if(cur->layer_activation_detail == RELU_ACT || cur->layer_activation_detail == OUTPUT_ACT){
            init_matrix(&(cur->dense.W),2,HE,input_size,input_size,cur->dense.nodes);
            // printf("initialized HE\n");
        }
        else if(cur->layer_activation_detail == SIGMOID_ACT || cur->layer_activation_detail == SOFTMAX_ACT){
            init_matrix(&(cur->dense.W),2,XAVIER,input_size,input_size,cur->dense.nodes);
            // printf("initialized XAVIER\n");
        }
        init_matrix(&(cur->dense.b), 1, ZEROS, 0, cur->dense.nodes);
    }
    init_matrix(&(cur->out), 2, NONE, 0, cur->batchsize, cur->dense.nodes);
}

//void init_dense_with_loss(Layer* pre, Layer* cur, Matrix *t, bool init_weight){}
void init_batchnormalization(Layer* pre, Layer* cur, bool init_weight){
    int feature_num = (pre->out->size)/(pre->batchsize);

    delete_matrix(cur->batchnormalization.dbeta);
    cur->batchnormalization.dbeta = NULL;
    delete_matrix(cur->batchnormalization.dgamma);
    cur->batchnormalization.dgamma = NULL;
    delete_matrix(cur->batchnormalization.std);
    cur->batchnormalization.std = NULL;
    delete_matrix(cur->batchnormalization.xc);
    cur->batchnormalization.xc = NULL;
    delete_matrix(cur->batchnormalization.xn);
    cur->batchnormalization.xn = NULL;
    if(init_weight){
        delete_matrix(cur->batchnormalization.vbeta);
        cur->batchnormalization.vbeta = NULL;
        delete_matrix(cur->batchnormalization.vgamma);
        cur->batchnormalization.vgamma = NULL;
        delete_matrix(cur->batchnormalization.mbeta);
        cur->batchnormalization.mbeta = NULL;
        delete_matrix(cur->batchnormalization.mgamma);
        cur->batchnormalization.mgamma = NULL;
        init_matrix(&(cur->batchnormalization.gamma), 1, CUSTOM, 1.0, feature_num);
        init_matrix(&(cur->batchnormalization.beta), 1, CUSTOM, 0.0, feature_num);

        init_matrix(&(cur->batchnormalization.running_mean), 1, ZEROS, 0.0, feature_num);
        init_matrix(&(cur->batchnormalization.running_var), 1, ZEROS, 0.0, feature_num);
    }
    copy_matrix(pre->out, &(cur->out), true);
}

//void init_maxpooling1d(Layer* pre, Layer* cur);
void init_maxpooling2d(Layer* pre, Layer* cur){
    int out_h, out_w;
    int pad_h, pad_w;
    int pre_h = pre->out->shape[2];
    int pre_w = pre->out->shape[3];

    delete_matrix(cur->maxpooling2d.x);
    cur->maxpooling2d.x = NULL;
    // printf("max: exception\n");
    if(cur->maxpooling2d.padding_type == VALID){
        cur->maxpooling2d.padding_h_down = 0;
        cur->maxpooling2d.padding_h_up = 0;
        cur->maxpooling2d.padding_w_left = 0;
        cur->maxpooling2d.padding_w_right = 0;
    }
    else{   //VALID
        pad_h = (pre_h -1)*(cur->maxpooling2d.stride_h) + (cur->maxpooling2d.pool_h) - pre_h;
        pad_w = (pre_w -1)*(cur->maxpooling2d.stride_w) + (cur->maxpooling2d.pool_w) - pre_w;
        if(pad_h%2){  //odd
            cur->maxpooling2d.padding_h_up = pad_h/2 + 1;
            cur->maxpooling2d.padding_h_down = pad_h/2;
        }
        else{
            cur->maxpooling2d.padding_h_up = pad_h/2;
            cur->maxpooling2d.padding_h_down = pad_h/2;
        }
        if(pad_w%2){  //odd
            cur->maxpooling2d.padding_w_left = pad_w/2 + 1;
            cur->maxpooling2d.padding_w_right = pad_w/2;
        }
        else{
            cur->maxpooling2d.padding_w_left = pad_w/2;
            cur->maxpooling2d.padding_w_right = pad_w/2;
        }
    }
    // printf("max: exception\n");
    out_h = (pre_h+(cur->maxpooling2d.padding_h_up) + (cur->maxpooling2d.padding_h_down) - (cur->maxpooling2d.pool_h))/(cur->maxpooling2d.stride_h) + 1;
    out_w = (pre_w+(cur->maxpooling2d.padding_w_left) + (cur->maxpooling2d.padding_w_right) - (cur->maxpooling2d.pool_w))/(cur->maxpooling2d.stride_w) + 1;
    

    // printf("max: exception\n");
    init_matrix(&(cur->out), 4, NONE, 0, cur->batchsize, pre->out->shape[1], out_h, out_w);
    cur->maxpooling2d.arg_max_size = cur->out->size;
    if(cur->maxpooling2d.arg_max != NULL){
        free(cur->maxpooling2d.arg_max);
        cur->maxpooling2d.arg_max = NULL;
    }
    cur->maxpooling2d.arg_max = (int *)malloc(sizeof(int)*(cur->out->size));
}

void init_relu(Layer* pre, Layer* cur){
    if(cur->relu.mask != NULL){
        free(cur->relu.mask);
        cur->relu.mask = NULL;
    }
    cur->relu.mask = (bool*)malloc(sizeof(bool)*(pre->out->size));
    copy_matrix(pre->out, &(cur->out), true);
}

void init_sigmoid(Layer* pre, Layer* cur){
    copy_matrix(pre->out, &(cur->out), true);
}
//void init_softmax(Layer* pre, Layer* cur);
void init_softmax_with_loss(Layer* pre, Layer* cur){
    cur->softmax.loss = 0.0;
    init_matrix(&(cur->softmax.y), 2, NONE, 0, cur->batchsize, cur->softmax.nodes);
    cur->softmax.t = NULL;
}

void init_flatten(Layer* pre, Layer* cur){
    copy_matrix(pre->out, &(cur->out), true);
    reshape_matrix(cur->out, 2, cur->batchsize, (pre->out->size)/(cur->batchsize));
}

void init_dropout(Layer* pre, Layer* cur){
    if(cur->dropout.mask != NULL){
        free(cur->dropout.mask);
        cur->dropout.mask = NULL;
    }
    cur->dropout.mask = (bool*)malloc(sizeof(bool)*(pre->out->size));
    copy_matrix(pre->out, &(cur->out), true);
}

double* forward(Layer* model, Matrix *input, Matrix *t,int batchsize, bool train_flg){
    Layer* cur_layer = model;
    Layer* prev_layer = model->prev;
    double *loss_pt = NULL;

    // printf("forwarding...\n");
    while(cur_layer){
        // printf("forward\n");
        cur_layer->batchsize = batchsize;
        switch (cur_layer->layer_type)
        {
        case INPUT:
            // printf("forward: input\n");
            forward_input(cur_layer,input);
            // printf("%d, %d, %d, %d", cur_layer->out->shape[0],cur_layer->out->shape[1],cur_layer->out->shape[2],cur_layer->out->shape[3]);
            break;
        case DENSE:
            // printf("forward: dense\n");
            if(cur_layer->dense.loss_function == NONE)
                forward_dense(prev_layer, cur_layer);
            else{
                printf("not a constructed layer\n");
                exit(2);
            }
                //loss_pt = forward_dense_with_loss(prev_layer, cur_layer, t);
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            //forward_conv1d(prev_layer, cur_layer);
            break;
        case CONV2D:
            // printf("forward: conv2d\n");
            forward_conv2d(prev_layer, cur_layer);
            break;
        case MAXPOOLING1D:
            printf("not a constructed layer\n");
            exit(2);
            // forward_maxpooling1d(prev_layer, cur_layer);
            break;
        case MAXPOOLING2D:
            // printf("forward: max2d\n");
            forward_maxpooling2d(prev_layer, cur_layer);
            break;
        case BATCHNORM:
            // printf("forward: batchnorm\n");
            forward_batchnormalization(prev_layer, cur_layer,train_flg);
            // print_matrix(cur_layer->batchnormalization.xn, true);
            break;
        case RELU:
            // printf("forward: Relu\n");
            forward_relu(prev_layer, cur_layer);
            break;
        case SOFTMAX:
            // printf("forward: softmax\n");
            if(cur_layer->softmax.loss_function != NONE)
                loss_pt = forward_softmax_with_loss(prev_layer, cur_layer, t);
            else{
                printf("not a constructed layer\n");
                exit(2);
            }
            // printf("forward: softmax_end\n");
            //     forward_softmax(prev_layer, cur_layer);
            break;
        case SIGMOID:
            // printf("forward: sig\n");
            forward_sigmoid(prev_layer, cur_layer);
            break;
        case FLATTEN:
            // printf("forward: flatten\n");
            forward_flatten(prev_layer, cur_layer);
            break;
        case DROPOUT:
            // printf("forward: dropout\n");
            forward_dropout(prev_layer,cur_layer, train_flg);
            break;
        default:
            printf("wrong layer type\n");
            exit(2);
            break;
        }
        // print_matrix(cur_layer->out, false);
        prev_layer = cur_layer;
        cur_layer = cur_layer->next;
    }
    return loss_pt;
}

void forward_input(Layer* cur, Matrix* input){
    if((cur->out) != NULL && (cur->out->matrix) == NULL){
        delete_matrix(cur->out);
        cur->out = NULL;
    }
    cur->out = input;
}

//void forward_conv1d(Layer* pre, Layer* cur){}

void forward_conv2d(Layer* pre, Layer* cur){
    int FN = cur->conv2d.W->shape[0];
    int FH = cur->conv2d.W->shape[2];
    // int FW = cur->conv2d.W->shape[3];
    int N = pre->out->shape[0];
    // int H = pre->out->shape[2];
    // int W = pre->out->shape[3];
    int out_h = cur->out->shape[2];
    int out_w = cur->out->shape[3];
    Matrix *_out = NULL,*_col_W = NULL;
    //init
    // print_matrix(pre->out, true);
    
    im2col_matrix(pre->out, &(cur->conv2d.col), cur->conv2d.filter_size_h, cur->conv2d.filter_size_w, cur->conv2d.stride_h, cur->conv2d.stride_w,cur->conv2d.padding_h_up, cur->conv2d.padding_h_down, cur->conv2d.padding_w_left, cur->conv2d.padding_w_right);
    copy_matrix(cur->conv2d.W, &_col_W, false);
    reshape_matrix(_col_W, 2, FN, (cur->conv2d.W->size)/FN);
    transpose_matrix(_col_W, &(cur->conv2d.col_W), 1,0);
    delete_matrix(_col_W);

    dot2d_matrix(cur->conv2d.col, cur->conv2d.col_W, &(_out),1.0);
    reshape_matrix(_out, 4, N, out_h,out_w,FN);
    transpose_matrix(_out,&(cur->out),0,3,1,2);
    delete_matrix(_out);

    for(int n=0;n<FN;n++){
        int _b = cur->conv2d.b->matrix[n];
        for(int b=0;b<cur->batchsize;b++){
            for(int oh=0;oh < out_h;oh++){
                for(int ow = 0;ow<out_w;ow++){
                    cur->out->matrix4d[b][n][oh][ow]+= _b;
                }
            }
        }
    }
}

void forward_dense(Layer* pre, Layer* cur){
    //init
    copy_matrix(pre->out, &(cur->dense.x), false);
    reshape_matrix(cur->dense.x, 2, cur->batchsize, (cur->dense.x->size)/(cur->batchsize));

    dot2d_matrix(cur->dense.x, cur->dense.W, &(cur->out),1.0);
    for(int n=0;n<cur->dense.nodes;n++){
        int _b = cur->dense.b->matrix[n];
        for(int b=0;b<cur->batchsize;b++){
            cur->out->matrix2d[b][n]+= _b;
        }
    }
}

//void forward_dense_with_loss(Layer* pre, Layer* cur, Matrix *t){}

void forward_batchnormalization(Layer* pre, Layer* cur, bool train_flg){
    int N, D;
    Matrix *x = NULL, *var = NULL, *mu = NULL;
    Matrix *xc = NULL;
    Matrix *std = NULL;
    Matrix *xn = NULL;

    Matrix *r_mean = cur->batchnormalization.running_mean;
    Matrix *r_mean_temp1 = NULL, *r_mean_temp2 = NULL;
    Matrix *r_var = cur->batchnormalization.running_var;
    Matrix *r_var_temp1 = NULL, *r_var_temp2 = NULL;
    double momentum = cur->batchnormalization.momentum;

    Matrix *gamma = cur->batchnormalization.gamma;
    Matrix *beta = cur->batchnormalization.beta;

    N = cur->batchsize;
    D = (pre->out->size)/N;
    //init

    copy_matrix(pre->out, &x, false);
    reshape_matrix(x,2, N, D);
    
    if(train_flg){
        init_matrix(&mu, 1, ZEROS, 0, D);
        for(int i=0;i<D;i++){
            for(int n=0;n<N;n++){
                mu->matrix[i] += x->matrix2d[n][i];
            }
            mu->matrix[i] /= N;
        }
        
        copy_matrix(x,&(cur->batchnormalization.xc),false);
        xc = cur->batchnormalization.xc;
        for(int n=0;n<N;n++){
            for(int i=0; i<D;i++){
                xc->matrix2d[n][i] -= mu->matrix[i];
            }
        }

        init_matrix(&var, 1, ZEROS, 0, D);
        for(int i=0;i<D;i++){
            for(int n=0;n<N;n++){
                var->matrix[i] += pow(xc->matrix2d[n][i],2);
            }
            var->matrix[i] /= N;
        }
        
        copy_matrix(var, &(cur->batchnormalization.std), false);
        std = cur->batchnormalization.std;
        for(int i=0;i<D;i++){
            std->matrix[i] = sqrt(std->matrix[i] + 0.001);
        }
        
        copy_matrix(xc,&(cur->batchnormalization.xn),false);
        xn = cur->batchnormalization.xn;
        for(int n=0;n<N;n++){
            for(int i=0; i<D;i++){
                xn->matrix2d[n][i] /= std->matrix[i];
            }
        }

        mul_variable_matrix(r_mean, momentum, &r_mean_temp1);
        mul_variable_matrix(mu, 1.0-momentum, &r_mean_temp2);
        add_two_matrix(r_mean_temp1,r_mean_temp2,&(cur->batchnormalization.running_mean));

        mul_variable_matrix(r_var, momentum, &r_var_temp1);
        mul_variable_matrix(var, 1.0-momentum, &r_var_temp2);
        add_two_matrix(r_var_temp1,r_var_temp2,&(cur->batchnormalization.running_var));
        
        delete_matrix(r_mean_temp1);
        delete_matrix(r_mean_temp2);
        delete_matrix(r_var_temp1);
        delete_matrix(r_var_temp2);

        delete_matrix(var);
        delete_matrix(mu);
        
    }
    else{
        copy_matrix(x,&xc,false);
        for(int n=0;n<N;n++){
            for(int i=0; i<D;i++){
                xc->matrix2d[n][i] -= r_mean->matrix[i];
            }
        }
        copy_matrix(xc,&xn,false);
        for(int n=0;n<N;n++){
            for(int i=0; i<D;i++){
                xn->matrix2d[n][i] /= sqrt(r_var->matrix[i] + 0.001);
            }
        }
    }
    delete_matrix(x);
    
    reshape_matrix(cur->out,2, N, D);
    for(int n=0;n<N;n++){
        for(int i=0; i<D;i++){
            cur->out->matrix2d[n][i] = (gamma->matrix[i])*(xn->matrix2d[n][i]) + (beta->matrix[i]);
        }
    }
    if(!train_flg){
        delete_matrix(xc);
        delete_matrix(xn);
    }
    reshape_by_arr_matrix(cur->out, pre->out->dim, pre->out->shape);
}

//void forward_maxpooling1d(Layer* pre, Layer* cur){}
void forward_maxpooling2d(Layer* pre, Layer* cur){
    int N = pre->out->shape[0];
    // int H = pre->out->shape[2];
    // int W = pre->out->shape[3];
    int out_c = cur->out->shape[1];
    int out_h = cur->out->shape[2];
    int out_w = cur->out->shape[3];
    Matrix *_out = NULL, *col = NULL;
    int arg_max_size = cur->maxpooling2d.arg_max_size;
    int pool_size = (cur->maxpooling2d.pool_h)*(cur->maxpooling2d.pool_w);
    int max_index;
    double max_value;

    //init

    im2col_matrix(pre->out, &(col), cur->maxpooling2d.pool_h, cur->maxpooling2d.pool_w, cur->maxpooling2d.stride_h, cur->maxpooling2d.stride_w,cur->maxpooling2d.padding_h_up, cur->maxpooling2d.padding_h_down, cur->maxpooling2d.padding_w_left, cur->maxpooling2d.padding_w_right);
    reshape_matrix(col,2, arg_max_size,pool_size);


    init_matrix(&_out, 1, NONE, 0, arg_max_size);
    for(int n=0;n<arg_max_size;n++){
        max_value = col->matrix2d[n][0];
        max_index = 0;
        for(int idx=0;idx<pool_size;idx++){
            double temp = col->matrix2d[n][idx];
            if(max_value< temp){
                max_index = idx;
                max_value = temp;
            }
        }
        cur->maxpooling2d.arg_max[n] = max_index;
        _out->matrix[n] = max_value;
    }
    reshape_matrix(_out, 4, N,out_h,out_w, out_c);
    transpose_matrix(_out, &(cur->out), 0,3,1,2);
    delete_matrix(_out);
    delete_matrix(col);
}

void forward_relu(Layer* pre, Layer* cur){
    bool temp;
    double temp_pre;
    int size = (pre->out->size);
    for(int i=0;i<size;i++){
        temp_pre = pre->out->matrix[i];
        temp = temp_pre > 0;
        cur->relu.mask[i] = temp;
        cur->out->matrix[i] = temp*temp_pre;
    }
}

void forward_sigmoid(Layer* pre, Layer* cur){
    int size = (pre->out->size);

    for(int i=0;i<size;i++){
        cur->out->matrix[i] = sigmoid_(pre->out->matrix[i]);
    }
}
//void forward_softmax(Layer* pre, Layer* cur){}
double* forward_softmax_with_loss(Layer* pre, Layer* cur, Matrix *t){
    if(cur->softmax.loss_function == CROSS_ENTROPY_ERROR){
        cur->softmax.t = t;
        softmax_(pre->out, &(cur->softmax.y));
        
        cross_entropy_error(cur->softmax.y,t,&(cur->softmax.loss));
        // printf("%lf\n", cur->softmax.loss);
        return &(cur->softmax.loss);
    }
}

void forward_flatten(Layer* pre, Layer* cur){
    copy_matrix(pre->out,&(cur->out),false);
    reshape_matrix(cur->out, 2, cur->batchsize, (cur->out->size)/(cur->batchsize));
}

void forward_dropout(Layer* pre, Layer* cur, bool train_flg){
    if(train_flg){
        for(int i=0;i<(cur->out->size);i++){
            cur->dropout.mask[i] = ((double)rand() / RAND_MAX)>(cur->dropout.dropout_ratio);
            cur->out->matrix[i] = (cur->dropout.mask[i])*(pre->out->matrix[i]);
        }
    }
    else{
        for(int i=0;i<(cur->out->size);i++)
            cur->out->matrix[i] = (pre->out->matrix[i])*(1-(cur->dropout.dropout_ratio));
    }
}

void backward(Layer* backward_start){
    Layer* cur_layer = backward_start;
    Layer* next_layer = backward_start->next;
    Layer* prev_layer = backward_start->prev;
    
    while(cur_layer){
        clock_t start = clock();
        clock_t end;
        switch (cur_layer->layer_type)
        {
        case INPUT:
            break;
        case DENSE:
            // printf("dense backward\n");
            // print_matrix(cur_layer->prev->prev->prev->batchnormalization.xn,true);
            if(cur_layer->dense.loss_function == NONE)
                backward_dense(prev_layer, next_layer, cur_layer);
            else{
                printf("not a constructed layer\n");
                exit(2);
            }
            end = clock();
            // printf("dense: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
                //forward_dense_with_loss(prev_layer, cur_layer);
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            //forward_conv1d(prev_layer, next_layer, cur_layer);
            break;
        case CONV2D:
            // printf("CONV2D backward\n");
            backward_conv2d(prev_layer, next_layer, cur_layer);
            end = clock();
            // printf("conv2d: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
            break;
        case MAXPOOLING1D:
            printf("not a constructed layer\n");
            exit(2);
            // forward_maxpooling1d(prev_layer, next_layer, cur_layer);
            break;
        case MAXPOOLING2D:
            // printf("MAXPOOLING2D backward\n");
            backward_maxpooling2d(prev_layer, next_layer, cur_layer);
            end = clock();
            // printf("maxpooling: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
            break;
        case BATCHNORM:
            // printf("BATCHNORM backward\n");
            // print_matrix(cur_layer->batchnormalization.xn,true);
            backward_batchnormalization(prev_layer, next_layer, cur_layer);
            end = clock();
            // printf("batchnorm: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
            break;
        case RELU:
            // printf("RELU backward\n");
            // print_matrix(prev_layer->batchnormalization.xn,true);
            backward_relu(prev_layer, next_layer, cur_layer);
            end = clock();
            // printf("relu: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
            break;
        case SOFTMAX:
            // printf("SOFTMAX backward\n");
            // print_matrix(cur_layer->prev->prev->prev->prev->batchnormalization.xn,true);
            if(cur_layer->softmax.loss_function != NONE)
                backward_softmax_with_loss(prev_layer, cur_layer);
            else{
                printf("not a constructed layer\n");
                //     backward_softmax(prev_layer, next_layer, cur_layer);
                exit(2);
            }
            end = clock();
            // printf("softmax: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
            break;
        case SIGMOID:
            backward_sigmoid(prev_layer, next_layer, cur_layer);
            break;
        case FLATTEN:
            // printf("FLATTEN backward\n");
            backward_flatten(prev_layer, next_layer, cur_layer);
            break;
        case DROPOUT:
            // printf("DROPOUT backward\n");
            backward_dropout(prev_layer, next_layer,cur_layer);
            break;
        default:
            printf("wrong layer type\n");
            exit(2);
            break;
        }
        next_layer = cur_layer;
        cur_layer = cur_layer->prev;
        if(cur_layer){
            prev_layer = cur_layer->prev;
        }
        
    }
    
}

//void backward_conv1d(Layer* pre,Layer* next, Layer* cur){}

void backward_conv2d(Layer* pre, Layer* next, Layer* cur){
    int FN = cur->conv2d.W->shape[0];
    int C = cur->conv2d.W->shape[1];
    int FH = cur->conv2d.W->shape[2];
    int FW = cur->conv2d.W->shape[3];
    Matrix *dout = NULL, *dcol = NULL, *dWT = NULL;
    clock_t start = clock();
    transpose_matrix(next->dout, &dout, 0,2,3,1);
    // printf("transpose4d: %lf\n", (double)(clock()-start)/CLOCKS_PER_SEC);
    
    reshape_matrix(dout, 2, (dout->size)/FN,FN);
    //db
    init_matrix(&(cur->conv2d.db),1, ZEROS, 0, FN);
    for(int fn=0;fn<FN;fn++){
        for(int i=0;i<(dout->size)/FN;i++){
            cur->conv2d.db->matrix[fn] += dout->matrix2d[i][fn];
        }
    }
    
    //dW
    // transpose_matrix(cur->conv2d.col, &colT, 1,0);
    dot2d_TN_matrix(cur->conv2d.col, dout, &dWT,1.0);
    transpose_matrix(dWT, &(cur->conv2d.dW), 1,0);
    reshape_matrix(cur->conv2d.dW,4, FN,C,FH,FW);
    // delete_matrix(colT);
    delete_matrix(dWT);
    
    //dx (pre dout)
    // transpose_matrix(cur->conv2d.col_W,&colWT,1,0);
    dot2d_NT_matrix(dout, cur->conv2d.col_W, &dcol, 1.0);
    start = clock();
    col2im_matrix(pre->out, dcol, &(cur->dout), FH, FW, cur->conv2d.stride_h, cur->conv2d.stride_w,cur->conv2d.padding_h_up, cur->conv2d.padding_h_down, cur->conv2d.padding_w_left, cur->conv2d.padding_w_right);
    // printf("col2im: %lf\n", (double)(clock()-start)/CLOCKS_PER_SEC);
    delete_matrix(dcol);
    // delete_matrix(colWT);
    delete_matrix(dout);
}

void backward_dense(Layer* pre,Layer* next, Layer* cur){
    Matrix *dout = next->dout;
    // Matrix *WT = NULL, *xT = NULL;

    //dx (prev dout)
    // transpose_matrix(cur->dense.W, &WT, 1,0);
    dot2d_NT_matrix(dout, cur->dense.W, &(cur->dout),1.0);
    reshape_by_arr_matrix(cur->dout, pre->out->dim, pre->out->shape);
    // delete_matrix(WT);
    
    //dW
    // transpose_matrix(cur->dense.x, &xT, 1,0);
    dot2d_TN_matrix(cur->dense.x, dout, &(cur->dense.dW),1.0);
    // delete_matrix(xT);
    
    //db
    // print_matrix(cur->dense.db, false);
    init_matrix(&(cur->dense.db), 1, ZEROS, 0, cur->dense.nodes);
    // print_matrix(dout, true);
    // print_matrix(cur->dense.db, true);
    // printf("%d, %d\n", dout->size, cur->dense.db->size);
    for(int f = 0; f< cur->dense.nodes; f++){
        for(int b=0;b<cur->batchsize;b++){
            cur->dense.db->matrix[f] += dout->matrix2d[b][f];
        }
    }
    // print_matrix(cur->dense.db, false);
}

//void backward_dense_with_loss(Layer* pre,Layer* cur){}

void backward_batchnormalization(Layer* pre, Layer* next, Layer* cur){
    int batchsize = cur->batchsize;
    int feature = (next->dout->size)/(cur->batchsize);
    Matrix *dout = NULL, *xnXdout = NULL, *dxn = NULL, *dxc = NULL, *dxnXxc = NULL, *std2 = NULL,*dstd = NULL,*dvar = NULL, *dmu = NULL;
    
    copy_matrix(next->dout, &dout, false);
    if((dout->dim) !=2){
        reshape_matrix(dout, 2, batchsize, feature);
    }
    //dbeta
    init_matrix(&(cur->batchnormalization.dbeta), 1, ZEROS, 0, feature);
    for(int f = 0; f< feature; f++){
        for(int b=0;b<batchsize;b++){
            cur->batchnormalization.dbeta->matrix[f] += dout->matrix2d[b][f];
        }
    }
    
    //dgamma
    mul_two_matrix(cur->batchnormalization.xn, dout, &xnXdout);
    init_matrix(&(cur->batchnormalization.dgamma), 1, ZEROS, 0, feature);
    for(int f = 0; f< feature; f++){
        for(int b=0;b<batchsize;b++){
            cur->batchnormalization.dgamma->matrix[f] += xnXdout->matrix2d[b][f];
        }
    }
    delete_matrix(xnXdout);

    //dxn
    copy_matrix(dout,&dxn, false);
    for(int f = 0; f< feature; f++){
        double temp = cur->batchnormalization.gamma->matrix[f];
        for(int b=0;b<batchsize;b++){
            dxn->matrix2d[b][f] += temp;
        }
    }
    delete_matrix(dout);
    
    //dxc
    copy_matrix(dxn,&dxc, false);
    for(int f = 0; f< feature; f++){
        double temp = cur->batchnormalization.std->matrix[f];
        for(int b=0;b<batchsize;b++){
            dxc->matrix2d[b][f] /= temp;
        }
    }
    
    //dstd * 0.5
    // printf("error log\n");
    mul_two_matrix(dxn, cur->batchnormalization.xc, &dxnXxc);
    mul_two_matrix(cur->batchnormalization.std,cur->batchnormalization.std, &std2);
    for(int f = 0; f< feature; f++){
        double temp = std2->matrix[f];
        for(int b=0;b<batchsize;b++){
            dxnXxc->matrix2d[b][f] /= temp;
        }
    }
    delete_matrix(std2);
    delete_matrix(dxn);
    init_matrix(&dstd, 1, ZEROS, 0 , feature);
    for(int f = 0; f< feature; f++){
        for(int b=0;b<batchsize;b++){
            dstd->matrix[f] += dxnXxc->matrix2d[b][f];
        }
        dstd->matrix[f] *= -0.5;
    }
    delete_matrix(dxnXxc);
    //dvar
    div_two_matrix(dstd, cur->batchnormalization.std, &dvar);
    delete_matrix(dstd);
    //dxc2
    for(int f = 0; f< feature; f++){
        double temp = dvar->matrix[f];
        for(int b=0;b<batchsize;b++){
            dxc->matrix2d[b][f] += (2.0/ (double)batchsize)*temp*(cur->batchnormalization.xc->matrix2d[b][f]);
        }
    }
    delete_matrix(dvar);
    //dmu
    init_matrix(&dmu, 1, ZEROS, 0, feature);
    for(int f = 0; f< feature; f++){
        for(int b=0;b<batchsize;b++){
            dmu->matrix[f] += dxc->matrix2d[b][f];
        }
    }
    //dx (prev dout)
    copy_matrix(dxc, &(cur->dout), false);
    for(int f = 0; f< feature; f++){
        double temp = dmu->matrix[f];
        for(int b=0;b<batchsize;b++){
            cur->dout->matrix2d[b][f] -= temp/ (double)batchsize;
        }
    }
    delete_matrix(dmu);
    delete_matrix(dxc);
    reshape_by_arr_matrix(cur->dout, pre->out->dim, pre->out->shape);
    // print_matrix(cur->dout, true);
    // printf("%d\n", cur->out->size);
}

//void backward_maxpooling1d(Layer* pre, Layer* next, Layer* cur){}

void backward_maxpooling2d(Layer* pre, Layer* next, Layer* cur){
    int pool_h = cur->maxpooling2d.pool_h;
    int pool_w = cur->maxpooling2d.pool_w;
    int pool_size = pool_h*pool_w;
    Matrix *dout = NULL, *dcol = NULL;//,*col_W, *dcol, *colWT,*colT, *dW

    transpose_matrix(next->dout, &dout, 0,2,3,1);

    //dcol
    init_matrix(&dcol, 2, ZEROS, 0, dout->size, pool_size);
    for(int ds = 0; ds< (dout->size);ds++){
        dcol->matrix2d[ds][cur->maxpooling2d.arg_max[ds]] = dout->matrix[ds];
    }
    reshape_matrix(dcol, 2, (dout->shape[0])*(dout->shape[1])*(dout->shape[2]),(dout->shape[3])* pool_size);

    //dx (prev dout)
    col2im_matrix(pre->out, dcol, &(cur->dout), pool_h, pool_w, cur->maxpooling2d.stride_h, cur->maxpooling2d.stride_w,cur->maxpooling2d.padding_h_up, cur->maxpooling2d.padding_h_down, cur->maxpooling2d.padding_w_left, cur->maxpooling2d.padding_w_right);
    delete_matrix(dout);
    delete_matrix(dcol);
}

void backward_relu(Layer* pre, Layer* next, Layer* cur){
    copy_matrix(next->dout, &(cur->dout), false);
    for(int i=0;i<(cur->dout->size);i++){
        cur->dout->matrix[i] *= cur->relu.mask[i]; 
    }
}

void backward_sigmoid(Layer* pre, Layer* next, Layer* cur){
    Matrix * dout = next->dout;

    copy_matrix(dout, &(cur->dout), true);
    for(int i=0;i<(cur->dout->size);i++){
        cur->dout->matrix[i] = (dout->matrix[i])*(1.0-(cur->out->matrix[i]))*(cur->out->matrix[i]);
    }
}

// void backward_softmax(Layer* pre, Layer* next, Layer* cur){}

void backward_softmax_with_loss(Layer* pre, Layer* cur){
    int batchsize = cur->batchsize;
    Matrix *t = cur->softmax.t;
    Matrix *y = cur->softmax.y;
    Matrix *_t = NULL, *temp = NULL;
    
    //if categorical
    if(t->size != y->size){
        one_hot_encoding(t,&_t, 0, (cur->softmax.nodes) -1);
        sub_two_matrix(y,_t,&temp);
    }
    else{
        sub_two_matrix(y,t,&temp);
    }
    
    div_variable_matrix(temp,batchsize,&(cur->dout));
    delete_matrix(_t);
    delete_matrix(temp);
}

void backward_flatten(Layer* pre, Layer* next, Layer* cur){
    copy_matrix(next->dout, &(cur->dout),false);
    reshape_by_arr_matrix(cur->dout, pre->out->dim, pre->out->shape);
}

void backward_dropout(Layer* pre, Layer* next, Layer* cur){
    copy_matrix(next->dout, &(cur->dout), false);
    for(int i=0;i<(cur->dout->size);i++){
        cur->dout->matrix[i] *= cur->dropout.mask[i]; 
    }
}

// make_empty model
void make_empty_model(Layer* model, bool without_weight){
    Layer* cur_layer = model;
    
    while(cur_layer){
        cur_layer->batchsize = 0;
        
        switch (cur_layer->layer_type)
        {
        case INPUT:
            // printf("input seg\n");
            make_empty_input(cur_layer);
            
            break;
        case DENSE:
            // printf("dense seg\n");
            if(cur_layer->dense.loss_function == NONE)
                make_empty_dense(cur_layer, without_weight);
            else{
                //make_empty_dense_with_loss(cur_layer,without_weight);
                printf("not a constructed layer\n");
                exit(2);
            }
            break;
        case CONV1D:
            //make_empty_conv1d(cur_layer, without_weight);
            printf("not a constructed layer\n");
            exit(2);
            break;
        case CONV2D:
            // printf("conv2d seg\n");
            make_empty_conv2d(cur_layer, without_weight);
            break;
        case MAXPOOLING1D:
            //make_empty_maxpooling1d(cur_layer);
            printf("not a constructed layer\n");
            exit(2);
            break;
        case MAXPOOLING2D:
            // printf("max2d seg\n");
            make_empty_maxpooling2d(cur_layer);
            break;
        case BATCHNORM:
            // printf("bn seg\n");
            make_empty_batchnormalization(cur_layer, without_weight);
            break;
        case RELU:
            // printf("relu seg\n");
            make_empty_relu(cur_layer);
            break;
        case SOFTMAX:
            // printf("soft seg\n");
            if((cur_layer->softmax.loss_function) != NONE)
                make_empty_softmax_with_loss(cur_layer);
            else{
                //make_empty_softmax(cur_layer)
                printf("not a constructed layer\n");
                exit(2);
            }
            break;
        case SIGMOID:
            make_empty_sigmoid(cur_layer);
            break;
        case FLATTEN:
            make_empty_flatten(cur_layer);
            break;
        case DROPOUT:
            make_empty_dropout(cur_layer);
            break;
        default:
            printf("wrong layer type\n");
            exit(2);
            break;
        }
        cur_layer = cur_layer->next;
    }
}

void make_empty_input(Layer* cur){
    if(cur->out != NULL && cur->out->matrix == NULL){
        free(cur->out);
        cur->out = NULL;
    }
    else
        cur->out = NULL;
}

//void make_empty_conv1d(Layer* cur, bool init_weight){}

void make_empty_conv2d(Layer* cur, bool without_weight){
    delete_matrix(cur->conv2d.col);
    cur->conv2d.col = NULL;
    delete_matrix(cur->conv2d.col_W);
    cur->conv2d.col_W = NULL;
    delete_matrix(cur->conv2d.db);
    cur->conv2d.db = NULL;
    delete_matrix(cur->conv2d.dW);
    cur->conv2d.dW = NULL;

    if(!without_weight){
        delete_matrix(cur->conv2d.W);
        cur->conv2d.W = NULL;
        delete_matrix(cur->conv2d.b); 
        cur->conv2d.b = NULL; 
    }
    delete_matrix(cur->conv2d.mb);
    cur->conv2d.mb = NULL;
    delete_matrix(cur->conv2d.vb);
    cur->conv2d.vb = NULL;
    delete_matrix(cur->conv2d.mW);
    cur->conv2d.mW = NULL;
    delete_matrix(cur->conv2d.vW);
    cur->conv2d.vW = NULL;

    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

void make_empty_dense(Layer* cur, bool without_weight){
    delete_matrix(cur->dense.db);
    cur->dense.db = NULL;
    delete_matrix(cur->dense.dW);
    cur->dense.dW = NULL;
    delete_matrix(cur->dense.x);
    cur->dense.x = NULL;
    if(!without_weight){
        delete_matrix(cur->dense.W);
        cur->dense.W = NULL;
        delete_matrix(cur->dense.b); 
        cur->dense.b = NULL;
    }
    delete_matrix(cur->dense.mb);
    cur->dense.mb = NULL;
    delete_matrix(cur->dense.vb);
    cur->dense.vb = NULL;
    delete_matrix(cur->dense.mW);
    cur->dense.mW = NULL;
    delete_matrix(cur->dense.vW);
    cur->dense.vW = NULL;

    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

//void make_empty_dense_with_loss(Layer* cur, bool without_weight){}

void make_empty_batchnormalization(Layer* cur, bool without_weight){
    delete_matrix(cur->batchnormalization.dbeta);
    cur->batchnormalization.dbeta = NULL;
    delete_matrix(cur->batchnormalization.dgamma);
    cur->batchnormalization.dgamma = NULL;
    delete_matrix(cur->batchnormalization.std);
    cur->batchnormalization.std = NULL;
    delete_matrix(cur->batchnormalization.xc);
    cur->batchnormalization.xc = NULL;
    delete_matrix(cur->batchnormalization.xn);
    cur->batchnormalization.xn = NULL;

    if(!without_weight){
        delete_matrix(cur->batchnormalization.gamma);
        cur->batchnormalization.gamma = NULL;
        delete_matrix(cur->batchnormalization.beta);
        cur->batchnormalization.beta = NULL;
        
        delete_matrix(cur->batchnormalization.running_mean);
        cur->batchnormalization.running_mean = NULL;
        delete_matrix(cur->batchnormalization.running_var);
        cur->batchnormalization.running_var = NULL;
    }
    delete_matrix(cur->batchnormalization.vbeta);
    cur->batchnormalization.vbeta = NULL;
    delete_matrix(cur->batchnormalization.vgamma);
    cur->batchnormalization.vgamma = NULL;
    delete_matrix(cur->batchnormalization.mbeta);
    cur->batchnormalization.mbeta = NULL;
    delete_matrix(cur->batchnormalization.mgamma);
    cur->batchnormalization.mgamma = NULL;

    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

//void make_empty_maxpooling1d(Layer* cur){}

void make_empty_maxpooling2d(Layer* cur){
    delete_matrix(cur->maxpooling2d.x);
    cur->maxpooling2d.x = NULL;
    
    delete_matrix(cur->out);
    cur->out = NULL;

    cur->maxpooling2d.arg_max_size = 0;
    if(cur->maxpooling2d.arg_max != NULL){
        free(cur->maxpooling2d.arg_max);
        cur->maxpooling2d.arg_max = NULL;
    }
    delete_matrix(cur->dout);
    cur->dout =NULL;
}

void make_empty_relu(Layer* cur){
    if(cur->relu.mask != NULL){
        free(cur->relu.mask);
        cur->relu.mask = NULL;
    }
    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

void make_empty_sigmoid(Layer* cur){
    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

//void make_empty_softmax(Layer* cur){}

void make_empty_softmax_with_loss(Layer* cur){
    cur->softmax.t = NULL;
    delete_matrix(cur->softmax.y);
    cur->softmax.y = NULL;

    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

void make_empty_flatten(Layer* cur){
    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

void make_empty_dropout(Layer* cur){
    if(cur->dropout.mask != NULL){
        free(cur->dropout.mask);
        cur->dropout.mask = NULL;
    }
    delete_matrix(cur->out);
    cur->out = NULL;
    delete_matrix(cur->dout);
    cur->dout = NULL;
}

void delete_model(Layer* model){
    Layer *cur_layer = model;
    Layer *next_layer = NULL;

    make_empty_model(model, false);

    while(cur_layer){
        next_layer = cur_layer->next;
        free(cur_layer);
        cur_layer = next_layer;
    }
}

//regularization
void regularization(Layer* model, double *loss,int regularization_type, double lambda){
    Layer* cur_layer = model;
    double weight_decay = 0.0;
    Matrix *W = NULL;

    
    if(regularization_type == NONE)
        return;
    if(lambda == DEFAULT){
        lambda = 0.01;
    }

    while(cur_layer){
        switch (cur_layer->layer_type)
        {
        case DENSE:
            if(cur_layer->dense.loss_function == NONE){
                W = cur_layer->dense.W;
            }
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            //W = cur_layer->conv1d.W;
            break;
        case CONV2D:
            W = cur_layer->conv2d.W;
            break;
        default:
            W = NULL;
            break;
        }
        if(W != NULL){
            if(regularization_type == L1){
                for(int i=0;i<W->size;i++){
                    weight_decay += fabs(W->matrix[i]);
                }
            }
            else if(regularization_type == L2){
                for(int i=0;i<W->size;i++){
                    weight_decay += pow(W->matrix[i],2);
                }
            }
        }
        cur_layer = cur_layer->next;
    }
    
    weight_decay *= lambda;
    if(regularization_type == L2){
        weight_decay *= 0.5;
    }
    
    *loss += weight_decay;
}

void regularization_weight(Layer* model, int regularization_type, double lambda){
    Layer* cur_layer = model;
    // double weight_decay = 0.0;
    Matrix *W = NULL, *dW = NULL;

    if(regularization_type == NONE)
        return;

    if(lambda == DEFAULT){
        lambda = 0.01;
    }

    while(cur_layer){
        switch (cur_layer->layer_type)
        {
        case DENSE:
            if(cur_layer->dense.loss_function == NONE){
                W = cur_layer->dense.W;
                dW = cur_layer->dense.dW;
            }
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            //W = cur_layer->conv1d.W;
            //dW = cur_layer->conv1d.dW;
            break;
        case CONV2D:
            W = cur_layer->conv2d.W;
            dW = cur_layer->conv2d.dW;
            break;
        default:
            W = NULL;
            break;
        }
        if(W != NULL){
            if(regularization_type == L1){
                for(int i=0;i<W->size;i++){
                    dW->matrix[i] += sign(W->matrix[i])*lambda;
                }
            }
            else if(regularization_type == L2){
                for(int i=0;i<W->size;i++){
                    dW->matrix[i] += (W->matrix[i])*lambda;
                }
            }
        }
        cur_layer = cur_layer->next;
    }
}

//optimizer
void init_adam_optimizer(Optimizer** opt, double _lr, double _beta1, double _beta2){   //default: 0.001, 0.9, 0.999
    double lr = _lr;
    double beta1 = _beta1;
    double beta2 = _beta2;
    

    *opt = (Optimizer*)malloc(sizeof(Optimizer));
    // printf("adam init\n");
    (*opt)->opt_name = ADAM;
    if(lr == DEFAULT)
        lr = 0.001;
    if(beta1 == DEFAULT)
        beta1 = 0.9;
    if(beta2 == DEFAULT)
        beta2 = 0.999;
    (*opt)->adam.lr = lr;
    (*opt)->adam.beta1 = beta1;
    (*opt)->adam.beta2 = beta2;
    (*opt)->adam.iter = 0;
    // printf("adam init end\n");
}

void init_adam_params(Layer* model){
    Layer* cur_layer = model;
    
    while(cur_layer){
        switch (cur_layer->layer_type)
        {
        case DENSE:
            if(cur_layer->dense.loss_function == NONE){
                // printf("dense adam init\n");
                copy_matrix(cur_layer->dense.W, &(cur_layer->dense.mW), true);
                copy_matrix(cur_layer->dense.W, &(cur_layer->dense.vW), true);
                copy_matrix(cur_layer->dense.b, &(cur_layer->dense.mb), true);
                copy_matrix(cur_layer->dense.b, &(cur_layer->dense.vb), true);
            }
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            // copy_matrix(cur_layer->conv1d.W, cur_layer->conv1d.mW, true);
            // copy_matrix(cur_layer->conv1d.W, cur_layer->conv1d.vW, true);
            // copy_matrix(cur_layer->conv1d.b, cur_layer->conv1d.mb, true);
            // copy_matrix(cur_layer->conv1d.b, cur_layer->conv1d.vb, true);
            break;
        case CONV2D:
            // printf("conv2d adam init\n");
            copy_matrix(cur_layer->conv2d.W, &(cur_layer->conv2d.mW), true);
            copy_matrix(cur_layer->conv2d.W, &(cur_layer->conv2d.vW), true);
            copy_matrix(cur_layer->conv2d.b, &(cur_layer->conv2d.mb), true);
            copy_matrix(cur_layer->conv2d.b, &(cur_layer->conv2d.vb), true);
            // printf("conv2d adam end\n");
            break;
        case BATCHNORM:
            // printf("batchnorm adam init\n");
            copy_matrix(cur_layer->batchnormalization.gamma, &(cur_layer->batchnormalization.mgamma), true);
            copy_matrix(cur_layer->batchnormalization.gamma, &(cur_layer->batchnormalization.vgamma), true);
            copy_matrix(cur_layer->batchnormalization.beta, &(cur_layer->batchnormalization.mbeta), true);
            copy_matrix(cur_layer->batchnormalization.beta, &(cur_layer->batchnormalization.vbeta), true);
            break;
        default:
            break;
        }
        cur_layer = cur_layer->next;
    }
}

void adam_update(Layer *model, Adam *adam_opt){
    double lr_t;
    double lr = adam_opt->lr;
    double beta1 = adam_opt->beta1;
    double beta2 = adam_opt->beta2;
    double iter;
    int W_size, b_size, gamma_size, beta_size;
    Layer* cur_layer = model;

    adam_opt->iter += 1;
    iter = adam_opt->iter;

    lr_t = lr*sqrt(1.0-pow(beta2,iter))/(1.0-pow(beta1,iter));

    while(cur_layer){
        switch (cur_layer->layer_type)
        {
        case DENSE:
            if(cur_layer->dense.loss_function == NONE){
                // printf("dense update\n");
                W_size = cur_layer->dense.W->size;
                b_size = cur_layer->dense.b->size;
                
                for(int i=0; i<W_size;i++){
                    cur_layer->dense.mW->matrix[i] += (1.0-beta1)*((cur_layer->dense.dW->matrix[i])-(cur_layer->dense.mW->matrix[i]));
                    cur_layer->dense.vW->matrix[i] += (1.0-beta2)*(pow(cur_layer->dense.dW->matrix[i],2)-(cur_layer->dense.vW->matrix[i]));
                    cur_layer->dense.W->matrix[i] -= lr_t * (cur_layer->dense.mW->matrix[i]) / (sqrt(cur_layer->dense.vW->matrix[i]) +MIN_NUM);
                }
                for(int i=0; i<b_size;i++){
                    cur_layer->dense.mb->matrix[i] += (1.0-beta1)*((cur_layer->dense.db->matrix[i])-(cur_layer->dense.mb->matrix[i]));
                    cur_layer->dense.vb->matrix[i] += (1.0-beta2)*(pow(cur_layer->dense.db->matrix[i],2)-(cur_layer->dense.vb->matrix[i]));
                    cur_layer->dense.b->matrix[i] -= lr_t * (cur_layer->dense.mb->matrix[i]) / (sqrt(cur_layer->dense.vb->matrix[i]) +MIN_NUM);
                }
            }
            break;
        case CONV1D:
            printf("not a constructed layer\n");
            exit(2);
            // int W_size = cur_layer->conv1d.W->size;
            // int b_size = cur_layer->conv1d.b->size;
            
            // for(int i=0; i<W_size;i++){
            //     cur_layer->conv1d.mW->matrix[i] += (1.0-beta1)*((cur_layer->conv1d.dW->matrix[i])-(cur_layer->conv1d.mW->matrix[i]));
            //     cur_layer->conv1d.vW->matrix[i] += (1.0-beta2)*(pow(cur_layer->conv1d.dW->matrix[i],2)-(cur_layer->conv1d.vW->matrix[i]));
            //     cur_layer->conv1d.W->matrix[i] -= lr_t * (cur_layer->conv1d.mW->matrix[i]) / (sqrt(cur_layer->conv1d.vW->matrix[i]) +MIN_NUM);
            // }
            // for(int i=0; i<b_size;i++){
            //     cur_layer->conv1d.mb->matrix[i] += (1.0-beta1)*((cur_layer->conv1d.db->matrix[i])-(cur_layer->conv1d.mb->matrix[i]));
            //     cur_layer->conv1d.vb->matrix[i] += (1.0-beta2)*(pow(cur_layer->conv1d.db->matrix[i],2)-(cur_layer->conv1d.vb->matrix[i]));
            //     cur_layer->conv1d.b->matrix[i] -= lr_t * (cur_layer->conv1d.mb->matrix[i]) / (sqrt(cur_layer->conv1d.vb->matrix[i]) +MIN_NUM);
            // }
            break;
        case CONV2D:
            // printf("conv2d update\n");
            W_size = cur_layer->conv2d.W->size;
            b_size = cur_layer->conv2d.b->size;
            
            for(int i=0; i<W_size;i++){
                cur_layer->conv2d.mW->matrix[i] += (1.0-beta1)*((cur_layer->conv2d.dW->matrix[i])-(cur_layer->conv2d.mW->matrix[i]));
                cur_layer->conv2d.vW->matrix[i] += (1.0-beta2)*(pow(cur_layer->conv2d.dW->matrix[i],2)-(cur_layer->conv2d.vW->matrix[i]));
                cur_layer->conv2d.W->matrix[i] -= lr_t * (cur_layer->conv2d.mW->matrix[i]) / (sqrt(cur_layer->conv2d.vW->matrix[i]) +MIN_NUM);
            }
            for(int i=0; i<b_size;i++){
                cur_layer->conv2d.mb->matrix[i] += (1.0-beta1)*((cur_layer->conv2d.db->matrix[i])-(cur_layer->conv2d.mb->matrix[i]));
                cur_layer->conv2d.vb->matrix[i] += (1.0-beta2)*(pow(cur_layer->conv2d.db->matrix[i],2)-(cur_layer->conv2d.vb->matrix[i]));
                cur_layer->conv2d.b->matrix[i] -= lr_t * (cur_layer->conv2d.mb->matrix[i]) / (sqrt(cur_layer->conv2d.vb->matrix[i]) +MIN_NUM);
            }
            break;
        case BATCHNORM:
            // printf("batchnorm update\n");
            gamma_size = cur_layer->batchnormalization.gamma->size;
            beta_size = cur_layer->batchnormalization.beta->size;
            
            for(int i=0; i<gamma_size;i++){
                cur_layer->batchnormalization.mgamma->matrix[i] += (1.0-beta1)*((cur_layer->batchnormalization.mgamma->matrix[i])-(cur_layer->batchnormalization.mgamma->matrix[i]));
                cur_layer->batchnormalization.vgamma->matrix[i] += (1.0-beta2)*(pow(cur_layer->batchnormalization.vgamma->matrix[i],2)-(cur_layer->batchnormalization.vgamma->matrix[i]));
                cur_layer->batchnormalization.gamma->matrix[i] -= lr_t * (cur_layer->batchnormalization.mgamma->matrix[i]) / (sqrt(cur_layer->batchnormalization.vgamma->matrix[i]) +MIN_NUM);
            }
            for(int i=0; i<beta_size;i++){
                cur_layer->batchnormalization.mbeta->matrix[i] += (1.0-beta1)*((cur_layer->batchnormalization.dbeta->matrix[i])-(cur_layer->batchnormalization.mbeta->matrix[i]));
                cur_layer->batchnormalization.vbeta->matrix[i] += (1.0-beta2)*(pow(cur_layer->batchnormalization.dbeta->matrix[i],2)-(cur_layer->batchnormalization.vbeta->matrix[i]));
                cur_layer->batchnormalization.beta->matrix[i] -= lr_t * (cur_layer->batchnormalization.mbeta->matrix[i]) / (sqrt(cur_layer->batchnormalization.vbeta->matrix[i]) +MIN_NUM);
            }
            break;
        default:
            break;
        }
        cur_layer = cur_layer->next;
    }
}

//accuracy
double classification_accuracy(Layer*model, Matrix *x, Matrix*_t, double **loss){
    // double acc;
    Layer* tail = model_tail(model);
    Matrix *_y = NULL, *_y_temp = NULL;
    Matrix *Y = NULL, *T = NULL;
    
    
    init_model(model, x->shape, x->dim, false);
    
    *loss = forward(model, x, _t,x->shape[0],false);
    if(tail->layer_type == SOFTMAX){
        _y = tail->softmax.y;
        if(_y->dim == 1){
            copy_matrix(_y, &_y_temp, false);
            reshape_matrix(_y_temp, 2, 1, _y_temp->size);
            argmax_2d_matrix(_y_temp, &Y);
            delete_matrix(_y_temp);
        }
        else{
            argmax_2d_matrix(_y, &Y);
        }

        if(_t->dim == 2){
            argmax_2d_matrix(_t, &T);
        }
        else{
            copy_matrix(_t,&T, false);
        }
    }
    else if(tail->layer_type == DENSE){
        printf("accuarcy error: not constructed layer\n");
        exit(1);
        // loss = tail->dense.loss;
    }
    else{
        printf("accuarcy error: invalid model tail\n");
        exit(1);
    }
    
    int correct = 0;
    // print_matrix(Y,false);
    for(int i=0;i<model->batchsize;i++){
        if((Y->matrix[i]) == (T->matrix[i]))
            correct++;
    }
    delete_matrix(Y);
    delete_matrix(T);
    
    return (double)correct/(double)(model->batchsize);
}

int** classification_accuracy_with_batch(Layer*model, Matrix *x, Matrix*_t, double *acc, double *loss, int batchsize, int* pred_num,bool save_prediction){
    // double acc;
    Layer* tail = model_tail(model);
    Matrix *_y = NULL, *_y_temp = NULL;
    Matrix *Y = NULL, *T = NULL;
    Matrix *batch_x = NULL, *batch_y = NULL;
    // double *temp_loss;
    double loss_sum = 0.0;
    int cor_sum = 0;
    int data_num = x->shape[0];
    int iter = 0;
    int _pred_num = data_num - data_num%batchsize;
    int **pred_result = NULL;
    int cur_result_index = 0;

    if(x->shape[0] <batchsize){
        printf("invalid batchsize for prediction\n");
        exit(0);
    }
    
    init_batch(&batch_x, &batch_y, x, _t, batchsize);
    

    init_model(model, batch_x->shape, batch_x->dim, false);
    
    if(save_prediction){
        *pred_num = _pred_num;
        pred_result = (int **)malloc(sizeof(int*)*_pred_num);
    
        for(int i=0; i<_pred_num;i++){
            pred_result[i] = (int *)malloc(sizeof(int)*2);
        }
    }

    for(iter=0;batchsize*(iter+1)<=data_num;iter++){
        get_batch_by_index(iter, batch_x, batch_y, x, _t, batchsize);
        // printf("seg1\n");
        loss_sum += *forward(model, batch_x, batch_y,batch_x->shape[0],false);
        // printf("seg3%lf, \n", *temp_loss);
        // loss_sum += *temp_loss;
        // printf("seg2\n");
        if(tail->layer_type == SOFTMAX){
            _y = tail->softmax.y;
            if(_y->dim == 1){
                copy_matrix(_y, &_y_temp, false);
                reshape_matrix(_y_temp, 2, 1, _y_temp->size);
                argmax_2d_matrix(_y_temp, &Y);
                delete_matrix(_y_temp);
                _y_temp = NULL;
            }
            else{
                argmax_2d_matrix(_y, &Y);
            }

            if(batch_y->dim == 2){
                argmax_2d_matrix(batch_y, &T);
            }
            else{
                copy_matrix(batch_y,&T, false);
            }
        }
        else if(tail->layer_type == DENSE){
            printf("accuarcy error: not constructed layer\n");
            exit(1);
            // loss = tail->dense.loss;
        }
        else{
            printf("accuarcy error: invalid model tail\n");
            exit(1);
        }
        
        int correct = 0;
        // print_matrix(Y,false);
        for(int i=0;i<model->batchsize;i++){
            if((Y->matrix[i]) == (T->matrix[i]))
                correct++;
            if(save_prediction){
                pred_result[cur_result_index][0] = (int)(T->matrix[i]);
                pred_result[cur_result_index][1] = (int)(Y->matrix[i]);
                cur_result_index++;
            }
        }
        delete_matrix(Y);
        delete_matrix(T);
        Y = NULL;
        T = NULL;
        cor_sum += correct;
    }
    delete_connection_matrix(batch_x);
    delete_connection_matrix(batch_y);
    free(batch_x);
    free(batch_y);
    // printf("seg1\n");
    *loss = loss_sum/(double)(_pred_num);
    // printf("seg2\n");
    *acc = (double)cor_sum/(double)(_pred_num);

    if(save_prediction)
        return pred_result;
    else
        return NULL;
}

//model training
double** model_fit(Layer * model, Matrix* x_train, Matrix *y_train, Matrix *x_val, Matrix *y_val, int epochs, int batchsize, Optimizer* opt,int regularization_type,double regularization_lambda, int verbose, bool train_shuffle, const char* model_file_path){
    int data_num = x_train->shape[0];
    int single_x_size = (x_train->size)/(data_num);
    int single_y_size = (y_train->size)/data_num;
    double*loss = NULL;
    double **history = NULL;
    double min_loss = 999999999999.0;
    Matrix *batch_train_x = NULL, *batch_train_y = NULL;

    history = (double **)malloc(sizeof(double*)*epochs);
    
    for(int i=0; i<epochs;i++){
        history[i] = (double *)malloc(sizeof(double)*2);
    }
    

    printf("model fit start\n");
    // batch_train_x = (Matrix*)malloc(sizeof(Matrix));
    // batch_train_y = (Matrix*)malloc(sizeof(Matrix));

    // batch_train_x->dim = x_train->dim;
    // batch_train_y->dim = y_train->dim;
    // for(int i=1; i<x_train->dim;i++){
    //     batch_train_x->shape[i] = x_train->shape[i];
    // }
    // for(int i=1; i<y_train->dim;i++){
    //     batch_train_y->shape[i] = y_train->shape[i];
    // }
    // batch_train_x->shape[0] = batchsize;
    // batch_train_y->shape[0] = batchsize;
    // batch_train_x->size = batchsize*single_x_size;
    // batch_train_y->size = batchsize*single_y_size;
    // batch_train_x->connection_ptr = NULL;
    // batch_train_y->connection_ptr = NULL;
    init_batch(&batch_train_x, &batch_train_y, x_train, y_train, batchsize);

    printf("batchsize : %d, epochs: %d\n", batchsize, epochs);
    printf("train_data_num: %d\n", data_num);
    //training epochs
    for(int ep =0; ep<epochs;ep++){
        init_model(model, batch_train_x->shape, batch_train_x->dim, false);
        //mini-batch training
        for(int i=0;batchsize*(i+1)<=data_num;i++){
            // batch_train_x->matrix = (x_train->matrix)+i*batchsize*single_x_size;
            // batch_train_y->matrix = (y_train->matrix)+i*batchsize*single_y_size;
            // make_connection_matrix(batch_train_x);
            // make_connection_matrix(batch_train_y);
            get_batch_by_index(i, batch_train_x, batch_train_y, x_train, y_train, batchsize);

            // print_matrix(batch_train_y,false);
            // printf("%lf, %lf\n", get_value(batch_train_x, 1,1,1,1), batch_train_x->matrix4d[1][1][1][1]);
            // printf("forward_start\n");
            loss = forward(model, batch_train_x, batch_train_y, batchsize, true);
            // printf("forward done\n");
            if(ep == 0&& i==0){
                if(opt->opt_name == ADAM){
                    init_adam_params(model);
                }
            }
            
            regularization(model, loss, regularization_type, regularization_lambda);
            // printf("backward_start\n");
            backward(model_tail(model));
            // printf("backward done\n");
            regularization_weight(model, regularization_type,regularization_lambda);
            // printf("update start\n");
            if(opt->opt_name == ADAM)
                adam_update(model, &(opt->adam));
            if(model_tail(model)->layer_type == SOFTMAX){
                if(verbose >= 2){
                    printf("iteration for %d Epoch : %d\n", ep+1,i+1);
                    printf("selected data: %d~%d\n",batchsize*i,batchsize*(i+1)-1);
                    printf("middle acc: %lf, loss: %lf\n", classification_accuracy(model, batch_train_x, batch_train_y, loss), *loss);
                }
            }
            // init_model(model, batch_train_x->shape, batch_train_x->dim, false);
            // exit(0);
            // printf("update done\n");
        }
        // printf("iter_ done\n");
        //1 epoch finished
        
        
        if(model_tail(model)->layer_type == SOFTMAX){
            double acc;
            acc = classification_accuracy(model, batch_train_x, batch_train_y, &loss);
            
            double loss_train = *loss;
            // printf("%lf", *loss);
            double acc_val, loss_val;
            
            //validation print log
            init_model(model, x_val->shape, x_val->dim, false);
            classification_accuracy_with_batch(model, x_val, y_val, &acc_val, &loss_val, batchsize, NULL,false);
            //acc_val = classification_accuracy(model, x_val, y_val, &loss_val);
            if(verbose >= 1){
                printf("%d Epochs : accuracy - %lf, loss - %lf, val_accuracy - %lf, val_loss - %lf\n\n",ep+1, acc, loss_train, acc_val,loss_val);
            }
            history[ep][0] = acc;
            history[ep][1] = acc_val;

            if(model_file_path){
                // printf("model_saving\n");
                if(min_loss >= loss_val){
                    save_model(model, model_file_path);
                    printf("best model saved to %s\n\n", model_file_path);
                    min_loss = loss_val;
                    // printf("save complete\n");
                }
            }
        }
    }
    
    make_empty_model(model, true);
    delete_connection_matrix(batch_train_x);
    delete_connection_matrix(batch_train_y);
    free(batch_train_x);
    free(batch_train_y);

    return history;
}

void save_model(Layer *model, const char* filename){
    Layer* cur_layer = model;
    Layer* input_layer = model;
    Layer* prev_layer = model->prev;
    // int count = 0;
    // printf("initializing model\n");
    char _filename[100];

    cJSON *model_json = cJSON_CreateArray();

    //fprintf(file, "%d",cur_layer->layer_type);
    while(cur_layer){
        cJSON *layer_json = cJSON_CreateObject();
        // char layer_name[100] = "";
        switch (cur_layer->layer_type)
        {
        case INPUT:
            prev_layer = cur_layer;
            cur_layer = cur_layer->next;
            cJSON_Delete(layer_json);
            layer_json = NULL;
            continue;
            break;
        case DENSE:
            // printf("savelog: dense\n");
            if(cur_layer->dense.loss_function == NONE){
                cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
                cJSON_AddNumberToObject(layer_json, "nodes",cur_layer->dense.nodes);
                cJSON_AddNumberToObject(layer_json, "loss_function",cur_layer->dense.loss_function);
                cJSON_AddMatrixToObject(layer_json, *(cur_layer->dense.W), "W");
                cJSON_AddMatrixToObject(layer_json, *(cur_layer->dense.b), "b");
                //fprintf(file, "%d %d\n",cur_layer->layer_type, cur_layer->dense.nodes);
            }
            else{
                cJSON_Delete(layer_json);
                printf("not a constructed layer\n");
                exit(2);
            }
                //forward_dense_with_loss(prev_layer, cur_layer, init_weight);
            break;
        case CONV1D:
            cJSON_Delete(layer_json);
            printf("not a constructed layer\n");
            exit(2);
            break;
        case CONV2D:
            // printf("savelog: conv2d\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            cJSON_AddNumberToObject(layer_json, "filter_num",cur_layer->conv2d.filter_num);
            cJSON_AddNumberToObject(layer_json, "filter_size_w",cur_layer->conv2d.filter_size_w);
            cJSON_AddNumberToObject(layer_json, "filter_size_h",cur_layer->conv2d.filter_size_h);
            cJSON_AddNumberToObject(layer_json, "padding_type",cur_layer->conv2d.padding_type);
            cJSON_AddNumberToObject(layer_json, "stride_w",cur_layer->conv2d.stride_w);
            cJSON_AddNumberToObject(layer_json, "stride_h",cur_layer->conv2d.stride_h);
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->conv2d.W), "W");
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->conv2d.b), "b");
            //fprintf(file, "%d %d %d %d %d %d %d\n",cur_layer->layer_type, cur_layer->conv2d.filter_num, 
            //cur_layer->conv2d.filter_size_w, cur_layer->conv2d.filter_size_h, cur_layer->conv2d.padding_type, 
            //cur_layer->conv2d.stride_w, cur_layer->conv2d.stride_h);

            break;
        case MAXPOOLING1D:
            cJSON_Delete(layer_json);
            printf("not a constructed layer\n");
            exit(2);
            break;
        case MAXPOOLING2D:
            // printf("savelog: max2d\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            cJSON_AddNumberToObject(layer_json, "pool_w",cur_layer->maxpooling2d.pool_w);
            cJSON_AddNumberToObject(layer_json, "pool_h",cur_layer->maxpooling2d.pool_h);
            cJSON_AddNumberToObject(layer_json, "padding_type",cur_layer->maxpooling2d.padding_type);
            cJSON_AddNumberToObject(layer_json, "stride_w",cur_layer->maxpooling2d.stride_w);
            cJSON_AddNumberToObject(layer_json, "stride_h",cur_layer->maxpooling2d.stride_h);
            //fprintf(file, "%d %d %d %d %d %d\n",cur_layer->layer_type, 
            //cur_layer->maxpooling2d.pool_w, cur_layer->maxpooling2d.pool_h, 
            //cur_layer->maxpooling2d.padding_type, cur_layer->maxpooling2d.stride_w, cur_layer->maxpooling2d.stride_h);
            break;
        case BATCHNORM:
            // printf("savelog: batchnorm\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            cJSON_AddNumberToObject(layer_json, "momentum",cur_layer->batchnormalization.momentum);
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->batchnormalization.gamma), "gamma");
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->batchnormalization.beta), "beta");
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->batchnormalization.running_mean), "running_mean");
            cJSON_AddMatrixToObject(layer_json, *(cur_layer->batchnormalization.running_var), "running_var");
            //fprintf(file, "%d %lf\n",cur_layer->layer_type, cur_layer->batchnormalization.momentum);
            break;
        case RELU:
            // printf("savelog: relu\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            // fprintf(file, "%d\n",cur_layer->layer_type);
            break;
        case SOFTMAX:
            // printf("initlog: softmax\n");
            if((cur_layer->softmax.loss_function) != NONE){
                // printf("savelog: soft\n");
                cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
                cJSON_AddNumberToObject(layer_json, "nodes",cur_layer->softmax.nodes);
                cJSON_AddNumberToObject(layer_json, "loss_function",cur_layer->softmax.loss_function);
                //fprintf(file, "%d %d %d\n",cur_layer->layer_type, cur_layer->softmax.nodes, cur_layer->softmax.loss_function);
            }
            else{
                cJSON_Delete(layer_json);
                printf("not a constructed layer\n");
                exit(2);
            }
            //     forward_softmax(prev_layer, cur_layer);
            break;
        case SIGMOID:
            // printf("savelog: sig\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            //fprintf(file, "%d\n",cur_layer->layer_type);
            break;
        case FLATTEN:
            // printf("savelog: flatten\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            //fprintf(file, "%d\n",cur_layer->layer_type);
            break;
        case DROPOUT:
            // printf("savelog: drop\n");
            cJSON_AddNumberToObject(layer_json, "layer_type",cur_layer->layer_type);
            cJSON_AddNumberToObject(layer_json, "dropout_ratio",cur_layer->dropout.dropout_ratio);
            //fprintf(file, "%d %lf\n",cur_layer->layer_type, cur_layer->dropout.dropout_ratio);
            break;
        default:
            cJSON_Delete(layer_json);
            printf("wrong layer type\n");
            exit(2);
            break;
        }
        prev_layer = cur_layer;
        cur_layer = cur_layer->next;
        // sprintf(layer_name,"layer%d", count);
        cJSON_AddItemToArray(model_json, layer_json);
        layer_json = NULL;
    }

    

    char *json_string = cJSON_Print(model_json);
    cJSON_Delete(model_json);
    // printf("before killed\n");
    if(json_string){
        sprintf(_filename,"%s.json", filename);
        FILE* file = fopen(_filename, "w");
        if (file == NULL) {
            printf("Error opening the file.\n");
            return;
        }
        else{
            fputs(json_string,file);
            fclose(file);
        }
        
        cJSON_free(json_string);
    }
    else{
        printf("Error converting JSON to string\n");
    }
}

Layer* load_model(const char* filepath){
    char _filename[100];
    Layer *model = NULL;

    FILE* file = fopen(filepath, "r");

    if (file) {
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        char *json_data = (char *)malloc(file_size + 1);
        if (json_data) {
            fread(json_data, 1, file_size, file);
            json_data[file_size] = '\0';

            cJSON *model_json = cJSON_Parse(json_data);

            if (model_json) {
                int model_size = cJSON_GetArraySize(model_json);
                int layer_num = 0;
                Layer *cur_layer = NULL;
                create_model(&model);

                for(layer_num=0;layer_num<model_size;layer_num++){
                    cJSON *layer_json = cJSON_GetArrayItem(model_json, layer_num);
                    int layer_type = cJSON_GetObjectItem(layer_json,"layer_type")->valueint;
                    switch (layer_type)
                    {
                    case INPUT:
                        break;
                    case DENSE:
                    {
                        // printf("loadlog: dense\n");
                        int loss_function = cJSON_GetObjectItem(layer_json, "loss_function")->valueint;
                        if(loss_function == NONE){
                            int nodes = cJSON_GetObjectItem(layer_json,"nodes")->valueint;
                            add_dense(model, nodes);
                            cur_layer = model_tail(model);
                            cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"W"),&(cur_layer->dense.W));
                            cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"b"),&(cur_layer->dense.b));
                        }
                        else{
                            cJSON_Delete(model_json);
                            printf("not a constructed layer\n");
                            exit(2);
                        }
                        break;
                    }
                    case CONV1D:
                        cJSON_Delete(model_json);
                        printf("not a constructed layer\n");
                        exit(2);
                        break;
                    case CONV2D:
                    {
                        // printf("loadlog: conv2d\n");
                        int filter_num = cJSON_GetObjectItem(layer_json,"filter_num")->valueint;
                        int filter_size_w = cJSON_GetObjectItem(layer_json,"filter_size_w")->valueint;
                        int filter_size_h = cJSON_GetObjectItem(layer_json,"filter_size_h")->valueint;
                        int padding_type = cJSON_GetObjectItem(layer_json,"padding_type")->valueint;
                        int stride_w = cJSON_GetObjectItem(layer_json,"stride_w")->valueint;
                        int stride_h = cJSON_GetObjectItem(layer_json,"stride_h")->valueint;
                        add_conv2d(model, filter_num, filter_size_w, filter_size_h, padding_type, stride_w, stride_h);
                        cur_layer = model_tail(model);
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"W"),&(cur_layer->conv2d.W));
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"b"),&(cur_layer->conv2d.b));

                        break;
                    }
                    case MAXPOOLING1D:
                        cJSON_Delete(model_json);
                        printf("not a constructed layer\n");
                        exit(2);
                        break;
                    case MAXPOOLING2D:
                    {
                        // printf("loadlog: max2d\n");
                        int pool_w = cJSON_GetObjectItem(layer_json,"pool_w")->valueint;
                        int pool_h = cJSON_GetObjectItem(layer_json,"pool_h")->valueint;
                        int padding_type2 = cJSON_GetObjectItem(layer_json,"padding_type")->valueint;
                        int stride_w2 = cJSON_GetObjectItem(layer_json,"stride_w")->valueint;
                        int stride_h2 = cJSON_GetObjectItem(layer_json,"stride_h")->valueint;
                        add_maxpooling2d(model, pool_w, pool_h, padding_type2, stride_w2, stride_h2);
                        break;
                    }
                    case BATCHNORM:
                    {
                        // printf("loadlog: batchnorm\n");
                        double momentum = cJSON_GetObjectItem(layer_json,"momentum")->valuedouble;
                        add_batchnormalization(model, momentum);
                        cur_layer = model_tail(model);
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"gamma"),&(cur_layer->batchnormalization.gamma));
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"beta"),&(cur_layer->batchnormalization.beta));
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"running_mean"),&(cur_layer->batchnormalization.running_mean));
                        cJSON_GetMatrix(cJSON_GetObjectItem(layer_json,"running_var"),&(cur_layer->batchnormalization.running_var));
                        break;
                    }
                    case RELU:
                        // printf("loadlog: relu\n");
                        add_relu(model);
                        break;
                    case SOFTMAX:
                    {
                        // printf("initlog: softmax\n");
                        int loss_function2 = cJSON_GetObjectItem(layer_json, "loss_function")->valueint;
                        if(loss_function2 != NONE){
                            // printf("loadlog: soft\n");
                            int nodes = cJSON_GetObjectItem(layer_json,"nodes")->valueint;
                            add_softmax_with_loss(model, nodes, loss_function2);
                        }
                        else{
                            cJSON_Delete(model_json);
                            printf("not a constructed layer\n");
                            exit(2);
                        }
                        //     forward_softmax(prev_layer, cur_layer);
                        break;
                    }
                    case SIGMOID:
                        // printf("loadlog: sig\n");
                        add_sigmoid(model);
                        break;
                    case FLATTEN:
                        // printf("loadlog: flatten\n");
                        add_flatten(model);
                        break;
                    case DROPOUT:
                    {
                        // printf("loadlog: drop\n");
                        double dropout_ratio = cJSON_GetObjectItem(layer_json,"dropout_ratio")->valuedouble;
                        add_dropout(model, dropout_ratio);
                        break;
                    }
                    default:
                        cJSON_Delete(model_json);
                        printf("wrong layer type\n");
                        exit(2);
                        break;
                    }
                }

                cJSON_Delete(model_json);
            } else {
                printf("Error parsing JSON.\n");
            }

            free(json_data);
        } else {
            printf("Error allocating memory for JSON data.\n");
        }

        fclose(file); 
    } else {
        printf("Error opening file for reading.\n");
    }
    return model;
}
