#include "utils.h"
#include "DL_module.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define MAX_BUFFER 20000

void main(){
    char data_path[] = "../../[20221115]DMRS/";
    char result_path[] = "./learning_rate_results/0.001/"; /*"./filter_size_results/", "./feature_map_results/", "./max_pooling_results/" */
    char preprocessing_methods[][50] = {"origin/", "sequence_scaling/", "power_averaging/"}; 

    /*layer number vairables*/
    // char layer_num[][50] = {"layer_num1/", "layer_num2/"};
    // char layer1_filter_list[][50] = {"filter_num4/", "filter_num8/", "filter_num16/"};
    // char layer2_filter_list[][50] = {"filter_num4-4/", "filter_num8-8/", "filter_num16-16/", "filter_num4-8/", "filter_num8-16/"};

    /*filter size variables*/
    // char filter_size_list[][50] = {"filter_size2-2/", "filter_size3-3/", "filter_size4-4/"};

    /*max pooling variables*/
    // char pooling_list[][50] = {"maxpoolingO/", "maxpoolingX/"};
    // char result_path = "./max_pooling_results/";
    // char pooling_list[][50] = {"O/", "X/"};
    // int filter_size = 0;

    /*imagification shape variables*/
    // char imagification_shape_list[][50] = {"image_shape6-24/", "image_shape8-18/", "image_shape9-16/", "image_shape16-9/", "image_shape18-8/", "image_shape24-6/"};

    char path[300] = "\0";

    //get_file
    //preprocessing
    int class = 8;
    int C = 2;
    int H = 8;
    int W = 18;
    int data = 160000;
    int single_data_size = C*H*W;
    int test = 10;
    int f_num_length = 0;
    int filter_size = 2;
    int feature_map_num = 0;
    int layer2_fst_feature_map_num = 0;
    int layer2_snd_feature_map_num = 0;

    // int val = 160000*2/10;
    // int train = 160000*8/10;
    Matrix *data_x = NULL, *data_y = NULL;
    Matrix *_train_x = NULL,*_train_y = NULL;
    Matrix *train_x = NULL, *train_y = NULL;
    Matrix *test_x = NULL, *test_y = NULL;
    Matrix *val_x = NULL, *val_y = NULL;
    FILE *file = NULL;
    char* token;
    int row_index = 0;
    int idx_x = 0;
    int idx_y = 0;
    double line[MAX_BUFFER];
    int epochs = 1;
    int batchsize = 256;
    char* metrics[20] = {"acc", "val_acc"};
    char* pred_results_col[20] = {"True", "Pred"};
    time_t start;
    time_t end;

    srand(time(NULL));

    printf("DL_C_START\n");
    time(&start);

    for(int p_num = 2; p_num < 3; p_num++){
        for(int d_num = 0; d_num < 1; d_num++){
            init_matrix(&data_x, 4, NONE, 0, data, C, H, W);
            // delete_matrix(data_x);
            // exit(0);
            init_matrix(&data_y, 1, NONE, 0, data);
            
            sprintf(path, "%s%s%d_SNR.csv", data_path, preprocessing_methods[p_num], d_num);
            
            file = fopen(path, "r");
            
            if (file == NULL) {
                printf("invalid file\n");
                exit(5);
            }
            fgets(line, sizeof(line), file);

            idx_x = 0;
            idx_y = 0;
            while (fgets(line, sizeof(line), file)) {
                row_index = 0;
                line[strcspn(line, "\n")] = '\0';
                token = strtok(line, ",");
                while (token != NULL) {
                    if(row_index%289 == 288){
                        data_y->matrix[idx_y] = atof(token);
                        idx_y++;
                    }
                    else{
                        data_x->matrix[idx_x] = atof(token);
                        idx_x++;
                    }
                    row_index++;
                    token = strtok(NULL, ",");
                }
            }
            fclose(file);
            // print_matrix(data_x,true);
            // print_matrix(data_y, true);

            train_test_split_matrix(data_x, data_y, &_train_x, &_train_y, &test_x,&test_y, 0.8, false);
            train_test_split_matrix(_train_x, _train_y, &train_x, &train_y, &val_x,&val_y, 0.8, false);
            
            delete_matrix(data_x);
            data_x = NULL;
            delete_matrix(data_y);
            data_y = NULL;
            delete_matrix(_train_x);
            _train_x = NULL;
            delete_matrix(_train_y);
            _train_y = NULL;

            print_matrix(train_x, true);
            print_matrix(train_y, true);
            print_matrix(test_x, true);
            print_matrix(test_y, true);
            print_matrix(val_x, true);
            print_matrix(val_y, true);
            // printf("%lf\n", val_x->matrix4d[0][1][3][4]);
            // exit(0);

            
            // //model initializing
            Layer *model;
            create_model(&model);
            add_conv2d(model, 16, filter_size, filter_size, SAME, 1, 1);
            // printf("conv2d added\n");
            add_batchnormalization(model, 0.99);
            // printf("batchnorm added\n");
            add_relu(model);
            // printf("relu added\n");

            add_flatten(model);
            // printf("flatten added\n");

            add_dense(model, 1024);
            // printf("dense added\n");
            add_batchnormalization(model, 0.99);
            add_relu(model);

            add_dropout(model, 0.25);

            add_dense(model, class);
            add_softmax_with_loss(model, class, CROSS_ENTROPY_ERROR);
            // printf("softmax added\n");

            
            printf("initializing model\n");
            init_model(model, train_x->shape, train_x->dim, true);
            printf("model_init\n");
            //opt
            Optimizer* adam;
            init_adam_optimizer(&adam, 0.001, 0.9, 0.999);
            // print_matrix(train_x, false);
            // printf("%d,%d,%d,%d,", train_x->shape[0], train_x->shape[1], train_x->shape[2], train_x->shape[3]);
            
            /*
                model path
            */
            

            sprintf(path,"%s%s%d_bestmodel", result_path, preprocessing_methods[p_num], d_num);
            
            /*
                model path end
            */

            double **history = model_fit(model, train_x, train_y, val_x, val_y, epochs, batchsize,adam, L2, 0.0001, 1, false, path);
            delete_model(model);
            model = NULL;
            free(adam);
            adam = NULL;
            

            sprintf(path,"%s%s%d_history.csv", result_path, preprocessing_methods[p_num], d_num);
            
            saveToCSV(path, epochs, 2, history, metrics);
            
            for(int h=0;h<epochs;h++){
                free(history[h]);
            }
            free(history);

            /*
                model path
            */
            sprintf(path,"%s%s%d_bestmodel.json", result_path, preprocessing_methods[p_num], d_num);
            /*
                model path end
            */


            model = load_model(path);
            // printf("%d", model->next->layer_type);
            // printf("seg1\n");
            
            init_model(model, test_x->shape, test_x->dim, false);
            // printf("seg2\n");
            double loss;
            int pred_num;
            double acc = 0;
            int **pred_matrix = NULL;
            // loss = forward(model, test_x, test_y, 256, false);
            pred_matrix = classification_accuracy_with_batch(model, test_x, test_y,&acc, &loss, batchsize, &pred_num,true);
            printf("LOADED MODEL TEST acc/loss: %lf/%lf\n", acc, loss);

            delete_model(model);

            /*
                model path
            */
            sprintf(path,"%s%s%d_prediction.csv", result_path, preprocessing_methods[p_num], d_num);
            /*
                model path end
            */            
            
            saveToCSV_int(path, pred_num, 2, pred_matrix, pred_results_col);
            
            for(int p=0;p<pred_num;p++){
                free(pred_matrix[p]);
            }
            free(pred_matrix);


            delete_matrix(train_x);
            train_x = NULL;
            delete_matrix(train_y);
            train_y = NULL;
            delete_matrix(val_x);
            val_x = NULL;
            delete_matrix(val_y);
            val_y = NULL;
            delete_matrix(test_x);
            test_x = NULL;
            delete_matrix(test_y);
            test_y = NULL;
            
            printf("\n\ntraining_test end %dSNR, %s\n\n", d_num, preprocessing_methods[p_num]);
        }
    }
    time(&end);
    printf("Taken time: %f\n", (float)(end-start));
}
