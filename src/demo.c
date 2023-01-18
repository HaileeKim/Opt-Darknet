#define _GNU_SOURCE
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "rtod.h"
#include "darknet.h"
#include "option_list.h"
#include "dark_cuda.h"

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#ifdef V4L2
#include "v4l2.h"
#endif

#ifdef OPENCV

#include "http_stream.h"
#include "j_header.h"

extern int buff_index=0;
extern int cnt = 0;
extern double cycle_array[QLEN] = {0,};
extern int ondemand = 1;
extern int num_object = 0;
extern int measure = 1;

extern int nboxes = 0;
extern detection *dets = NULL;

extern float fps = 0;
extern float demo_thresh = 0;
extern int demo_ext_output = 0;
extern long long int frame_id = 0;
extern int demo_json_port = -1;
extern int demo_index = 0;
extern int letter_box = 0;
extern int fetch_offset = 0; // zero slack


extern int classes;
extern int top;
extern int* indexes;

extern double e_fetch_sum = 0;
extern double b_fetch_sum = 0;
extern double d_fetch_sum = 0;
extern double e_infer_cpu_sum = 0;
extern double e_infer_gpu_sum = 0;
extern double d_infer_sum = 0;
extern double e_disp_sum = 0;
extern double b_disp_sum = 0;
extern double d_disp_sum = 0;
extern double slack_sum = 0;
extern double e2e_delay_sum = 0;
extern double fps_sum = 0;
extern double cycle_time_sum = 0;
extern double inter_frame_gap_sum = 0;
extern double num_object_sum = 0;
extern double transfer_delay_sum = 0;
extern double image_waiting_sum = 0;

extern float* predictions[NFRAMES];
extern float* prediction = 0;

static bool demo_skip_frame = false;
static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread = 0;
static volatile int run_detect_in_thread = 0;

int *fd_handler = NULL;
int c = 0;

layer l;

#if defined SYNC || defined ASYNC || defined TWO_STAGE
int mem_arr[3][2000]={0,};
int base_mTOT=0;
int base_mCPU=0;
int base_mGPU=0;
#endif

int contention_free = 1;

/* Save result in csv*/
int write_result(char *file_path)
{
    static int exist=0;
    FILE *fp;
    int tick = 0;

    fp=fopen(file_path,"w+");

    if (fp == NULL) 
    {
        /* make directory */
        while(!exist)
        {
            int result;

            usleep(10 * 1000);

            result = mkdir(MEASUREMENT_PATH, 0766);
            if(result == 0) { 
                exist = 1;

                fp=fopen(file_path,"w+");
            }

            if(tick == 100)
            {
                fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

                return -1;
            }
            else tick++;
        }
    }
    else printf("\nWrite output in %s\n", file_path); 

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s\n", "e_fetch", "b_fetch", "d_fetch",
            "e_infer", "b_infer", "d_infer", "e_disp", "b_disp", "d_disp",
            "slack", "e2e_delay", "fps", "c_sys", "d_tran", "IWT", "IFG", "n_obj");
    for(int i=0;i<OBJ_DET_CYCLE_IDX;i++)
    {
        e_fetch_sum += e_fetch_array[i];
        b_fetch_sum += b_fetch_array[i];
        d_fetch_sum += d_fetch_array[i];
        e_infer_cpu_sum += e_infer_cpu_array[i];
        e_infer_gpu_sum += e_infer_gpu_array[i];
        d_infer_sum += d_infer_array[i];
        e_disp_sum += e_disp_array[i];
        b_disp_sum += b_disp_array[i];
        d_disp_sum += d_disp_array[i];
        slack_sum += slack[i];
        e2e_delay_sum += e2e_delay[i];
        fps_sum += fps_array[i];
        cycle_time_sum += cycle_time_array[i];
	    transfer_delay_sum += transfer_delay_array[i];
        image_waiting_sum += image_waiting_array[i];
        inter_frame_gap_sum += (double)inter_frame_gap_array[i];
        num_object_sum += (double)num_object_array[i];

        fprintf(fp, "%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d\n", e_fetch_array[i], b_fetch_array[i],d_fetch_array[i], 
                e_infer_cpu_array[i], e_infer_gpu_array[i], d_infer_array[i], e_disp_array[i], b_disp_array[i], d_disp_array[i], 
                slack[i], e2e_delay[i], fps_array[i], cycle_time_array[i], transfer_delay_array[i], inter_frame_gap_array[i], num_object_array[i]);
    }
    fclose(fp);

    return 1;
}

void push_data(void)
{
    b_fetch_array[cnt - CYCLE_OFFSET] = b_fetch;
    e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    //e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    d_fetch_array[cnt - CYCLE_OFFSET] = d_fetch;
    inter_frame_gap_array[cnt - CYCLE_OFFSET] = inter_frame_gap;
    transfer_delay_array[cnt - CYCLE_OFFSET] = transfer_delay;
    image_waiting_array[cnt - CYCLE_OFFSET] = image_waiting_time;

    e_infer_cpu_array[cnt - CYCLE_OFFSET] = d_infer - e_infer_gpu;
    e_infer_gpu_array[cnt - CYCLE_OFFSET] = e_infer_gpu;
    d_infer_array[cnt - CYCLE_OFFSET] = d_infer;

    fps_array[cnt - CYCLE_OFFSET] = fps;
    cycle_time_array[cnt - CYCLE_OFFSET] = 1000./fps;

#ifdef V4L2
    e2e_delay[cnt - CYCLE_OFFSET] = end_disp - frame[display_index].frame_timestamp;
#else
    e2e_delay[cnt - CYCLE_OFFSET] = end_disp - start_loop[display_index];
#endif

    printf("end_disp : %f \n", end_disp);  
#ifdef V4L2
    printf("timestamp : %f \n", frame[display_index].frame_timestamp);
#else
    printf("start loop : %f \n", start_loop[display_index]);  
#endif    
    printf("timestamp : %f \n", frame[display_index].frame_timestamp);
    
    e_disp_array[cnt - CYCLE_OFFSET] = d_disp - b_disp;
    b_disp_array[cnt - CYCLE_OFFSET] = b_disp;
    d_disp_array[cnt - CYCLE_OFFSET] = d_disp;
    slack[cnt - CYCLE_OFFSET] = slack_time;
    num_object_array[cnt - CYCLE_OFFSET] = num_object;

    printf("latency: %f\n",e2e_delay[cnt - CYCLE_OFFSET]);
    printf("cnt : %d\n",cnt);

    return;
}

/* Timestamp in ms */
double get_time_in_ms(void)
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC,&time_after_boot);
    return (time_after_boot.tv_sec*1000+time_after_boot.tv_nsec*0.000001);
}

/* Check if On-demand capture */
int check_on_demand(void)
{
    int env_var_int;
    char *env_var;
    static int size;
    int on_demand;

#if (defined V4L2)
    env_var = getenv("V4L2_QLEN");
#else
    env_var = getenv("OPENCV_QLEN");
#endif

    if(env_var != NULL){
        env_var_int = atoi(env_var);
    }
    else {
        printf("Using DEFAULT V4L Queue Length\n");
        env_var_int = 4;
    }

    switch(env_var_int){
        case 0 :
            on_demand = 1;
            size = 1;
            break;
        case 1:
            on_demand = 0;
            size = 1;
            break;
        case 2:
            on_demand = 0;
            size = 2;
            break;
        case 3:
            on_demand = 0;
            size = 3;
            break;
        case 4:
            on_demand = 0;
            size = 4;
            break;
        default :
            on_demand = 0;
            size = 4;
    }

    return on_demand;
}

void *fetch_in_thread(void *ptr)
{
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_fetch_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            if (demo_skip_frame)
                consume_frame(cap);
            this_thread_yield();
        }
        start_fetch = get_time_in_ms();

        int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
        if (letter_box)
            in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        else{
        
#ifdef V4L2
	    if(-1 == capture_image(&frame[buff_index], *fd_handler))
	    {
		perror("Fail to capture image");
		exit(0);
            }
            letterbox_image_into(frame[buff_index].frame, net.w, net.h, frame[buff_index].resize_frame);
            if(!frame[buff_index].resize_frame.data){
                printf("Stream closed.\n");
                flag_exit = 1;
                //exit(EXIT_FAILURE);
                return 0;
            }
#else

            in_s = get_image_from_stream_resize_with_timestamp(cap, net.w, net.h, net.c, &in_img, dont_close_stream, &frame[buff_index]);
            if(!in_s.data)
            {
                printf("Stream closed.\n");
                flag_exit = 1;
                //exit(EXIT_FAILURE);
                return 0;
            }
#endif           
    
            custom_atomic_store_int(&run_fetch_in_thread, 0);
    	}
      	end_fetch = get_time_in_ms();

        image_waiting_time = frame[buff_index].frame_timestamp - start_fetch;
        image_waiting_time -= fetch_offset;

    

        inter_frame_gap = GET_IFG(frame[buff_index].frame_sequence, frame_sequence_tmp);

        if(cnt >= (CYCLE_OFFSET - 5)){
            d_fetch = end_fetch - start_fetch;
            b_fetch = frame[buff_index].select;
            e_fetch = d_fetch - b_fetch - fetch_offset;
        }

        transfer_delay = frame[buff_index].select - image_waiting_time ;
    }
    return 0;
}

void *fetch_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_fetch_in_thread, 1);
    while (custom_atomic_load_int(&run_fetch_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_detect_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

        start_infer = get_time_in_ms();
        
        layer l = net.layers[net.n - 1];
        
#ifdef V4L2
        float *X = frame[detect_index].resize_frame.data;
#else
        float *X = det_s.data;
#endif
        float *prediction = network_predict(net, X);

        double e_i_cpu = get_time_in_ms();
            
#ifdef DNN
        memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
        mean_arrays(predictions, NFRAMES, l.outputs, avg);
        l.output = avg;
#else
        if(net.hierarchy) hierarchy_predictions(prediction, net.outputs, net.hierarchy, 1);
        top_predictions(net, top, indexes);
#endif
 
        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + NFRAMES / 2 + 1) % NFRAMES];
        demo_index = (demo_index + 1) % NFRAMES;

#ifdef V4L2
        dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
        printf("oboxes = %d", nboxes);
#else
        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
#endif
        end_infer = get_time_in_ms();
        d_infer = end_infer - start_infer;
        
        custom_atomic_store_int(&run_detect_in_thread, 0);
    }

    return 0;
}

void *detect_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_detect_in_thread, 1);
    while (custom_atomic_load_int(&run_detect_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

#ifdef V4L2
void *display_in_thread(void *ptr)
{
#ifdef DNN
    int c = show_image_cv(frame[display_index].frame, "Demo");
#else
    int c = show_image_cv(frame[display_index].frame, "Classifier Demo");
#endif

    if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
    {
        flag_exit = 1;
    }
    return 0;
}
#endif

double get_wall_time()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}


#if defined ASYNC || defined TWO_STAGE
void printMEM()
{
    char cmd[1024];
    int nMemTot =0;
    int nMemAva =0;
    int nMapMem =0;
    int mCPU, mGPU, mTOT;

    sprintf(cmd, "/proc/meminfo");
    FILE *fp = fopen(cmd, "r");
    if(fp == NULL) return;
    while(fgets(cmd, 1024, fp) != NULL) {
        if(strstr(cmd, "MemTotal"))  {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMemTot = atoi(size);
        }
        else if(strstr(cmd, "MemAvailable")) {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMemAva = atoi(size);
        }
        else if(strstr(cmd, "NvMapMemUsed")) {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMapMem = atoi(size);
            break;
        }
    }
    fclose(fp);

    mTOT = (nMemTot - nMemAva);
    mGPU = nMapMem;
    mCPU = mTOT - mGPU;

    base_mTOT = mTOT/1024;
    base_mCPU = mCPU/1024;
    base_mGPU = mGPU/1024;

    printf("mTOT: %dkB (=%dMB)\n",mTOT,mTOT/1024);
    printf("mCPU: %dkB (=%dMB)\n",mCPU,mCPU/1024);
    printf("mGPU: %dkB (=%dMB)\n",mGPU,mGPU/1024);

}

void mem()
{
    char cmd[1024];
    int nMemTot =0;
    int nMemAva =0;
    int nMapMem =0;
    int mCPU, mGPU, mTOT;
    static int count =0;

    sprintf(cmd, "/proc/meminfo");
    FILE *fp = fopen(cmd, "r");
    if(fp == NULL) return;
    while(fgets(cmd, 1024, fp) != NULL) {
        if(strstr(cmd, "MemTotal"))  {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMemTot = atoi(size);
        }
        else if(strstr(cmd, "MemAvailable")) {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMemAva = atoi(size);
        }
        else if(strstr(cmd, "NvMapMemUsed")) {
            char t[32];
            char size[32];
            sscanf(cmd, "%s%s", t, size);
            nMapMem = atoi(size);
            break;
        }
    }
    fclose(fp);

    mTOT = (nMemTot - nMemAva);
    mGPU = nMapMem;
    mCPU = mTOT - mGPU;

    mem_arr[0][count]=mTOT/1024;
    mem_arr[1][count]=mCPU/1024;
    mem_arr[2][count]=mGPU/1024;
    count++;
    printf("mTOT: %dkB (=%dMB)\n",mTOT,mTOT/1024);
    printf("mCPU: %dkB (=%dMB)\n",mCPU,mCPU/1024);
    printf("mGPU: %dkB (=%dMB)\n",mGPU,mGPU/1024);

}
#endif


void demo(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, 
        int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
        int benchmark, int benchmark_layers, int w, int h, int cam_fps)
{
    letter_box = letter_box_in;
    in_img = det_img = show_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_alphabet = alphabet;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;

    int img_w = w;
    int img_h = h;
    int cam_frame_rate= cam_fps;
    
    char *pipeline = NULL;
    char *PIPELINE = NULL;
    
#if (defined VANILLA)
    pipeline="V";
    PIPELINE = "Vanilla";
#elif (defined CONTENTION_FREE) 
    pipeline = "CF";
    PIPELINE = "CONTENTION_FREE";
#elif (defined ASYNC) 
    pipeline = "ASYNC";
    PIPELINE = "ASDYNCHRONOUS";
#elif (defined TWO_STAGE) 
    pipeline = "2S";
    PIPELINE = "TWO STAGE";
#elif (defined OPTIMIZATION) 
    pipeline = "OPT";
    PIPELINE = "OPTIMIZATION";
#else
    fprintf(stderr, "ERROR: Set either VANILLA or CONTENTION_FREE or ASYNCHRONOUS or TWO STAGE or OPTIMIZATION in Makefile\n");
    exit(0);
#endif

    //define measurement file name    
    char file_path[256];
    
    //date
    struct tm* t;
    time_t base = time(NULL);
    t = localtime(&base);
    
    //network name
    char* network = malloc(strlen(cfgfile));;
    strncpy(network, cfgfile + 4, (strlen(cfgfile)-8));
    
    sprintf(file_path, "measure/22%d%d_%s_%s.csv", t->tm_mon+1, t->tm_mday, network, pipeline);
    
    if(filename)
    {
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
        printf("vide=============================\n");
    }
    else
    {
        printf("Webcam index: %d\n", cam_index);
        

#ifdef V4L2
        char cam_dev[256] = "/dev/video";
        char index[256];
        sprintf(index, "%d", cam_index);
        strcat(cam_dev, index);
        printf("cam dev : %s\n", cam_dev);

        fd_handler = open_device(cam_dev, cam_frame_rate, img_w, img_h);
        if(fd_handler ==  NULL)
        {
            perror("Couldn't connect to webcam.\n");
        }
#else
        cap = get_capture_webcam_with_prop(cam_index, img_w, img_h, cam_frame_rate);
        if (!cap) {
#ifdef WIN32
            printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
            perror("Couldn't connect to webcam.\n");
        }
#endif
        
    }

    int j;
    
#ifdef DNN
    printf("Demo\n");
    list *options = read_data_cfg(datacfg);
    int classes = option_find_int(options, "classes", 20);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    demo_names = names;
    demo_classes = classes;

    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
        
// "==========================SSSSSSSSSSSSSOOOOOOOOOOOOOOLLLLLLLLLLLLL"==============

#ifndef TWO_STAGE
        load_weights(&net, weightfile);
#else
#ifndef ONDEMAND_LOAD
        load_weights(&net, weightfile);
#endif
#endif //Not TWO_STAGE
    }
    
    net.benchmark_layers = benchmark_layers;
#ifndef ONDEMAND_LOAD
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
#endif //ONDEMAND_LOAD
    srand(2222222);

    //layer l = net.layers[net.n-1];
    l = net.layers[net.n-1];
    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < NFRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

// WHY??????
    int i;
    for (i = 0; i < net.n; ++i) 
    {
        layer lc = net.layers[i];
        if (lc.type == YOLO) {
            lc.mean_alpha = 1.0 / NFRAMES;
            l = lc;
        }
    }
//================

    if (l.classes != demo_classes) {
        printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }
    flag_exit = 0;
    
#else
    printf("Classifier Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);

#ifndef TWO_STAGE
        load_weights(&net, weightfile);
#else
#ifndef ONDEMAND_LOAD
        load_weights(&net, weightfile);
#endif
#endif //Not TWO_STAGE

    }
    net.benchmark_layers = benchmark_layers;
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    //layer l = net.layers[net.n-1];
    l = net.layers[net.n-1];
    
#ifndef ONDEMAND_LOAD
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
#endif //ONDEMAND_LOAD

    srand(2222222);

    classes = option_find_int(options, "classes", 2);
    top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    int **names = get_labels(name_list);

    demo_names = names;
    demo_classes = classes;
    indexes = (int*)xcalloc(top, sizeof(int));
#endif

    pthread_t fetch_thread;
    pthread_t inference_thread;
    if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
    if (custom_create_thread(&inference_thread, 0, detect_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
    
#ifndef V4L2
#ifdef CONTENTION_FREE
    ondemand = check_on_demand();

    if(ondemand != 1) { 
        fprintf(stderr, "ERROR : R-TOD needs on-demand capture.\n");
        exit(0);
    }
#endif
#endif
    printf("OBJECT DETECTOR INFORMATION:\n"
            "  Capture: \"On-demand capture\"\n"
            "  Pipeline architecture: \"%s\"\n",
            PIPELINE);
    
    printf("ondemand : %d\n", ondemand);
    
#ifdef V4L2
	if(-1 == capture_image(&frame[buff_index], *fd_handler))
	{
		perror("Fail to capture image");
		exit(0);
	}
    frame[0].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[1].frame = frame[0].frame;
    frame[1].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[2].frame = frame[0].frame;
    frame[2].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);
#else


    fetch_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    fetch_in_thread_sync(0); //fetch_in_thread(0);
    detect_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;


    printf("==============================");
    for (j = 0; j < NFRAMES / 2; ++j) {
        free_detections(dets, nboxes);
        fetch_in_thread_sync(0); //fetch_in_thread(0);
        detect_in_thread_sync(0); //fetch_in_thread(0);
        det_img = in_img;
        det_s = in_s;
    }
#endif

    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
#ifdef DNN
        create_window_cv("Demo", full_screen, 1352, 1013);
#else   
        create_window_cv("Classifier Demo", full_screen, 512, 512);
#endif
        //make_window("Demo", 1352, 1013, full_screen);
    }

    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);
#ifndef V4L2
        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);
#endif

        //'H', '2', '6', '4'
        //'D', 'I', 'V', 'X'
        //'M', 'J', 'P', 'G'
        //'M', 'P', '4', 'V'
        //'M', 'P', '4', '2'
        //'X', 'V', 'I', 'D'
        //'W', 'M', 'V', '2'
    }

    int send_http_post_once = 0;
    const double start_time_lim = get_time_point();
    double before = get_time_point();
    double before_1 = get_time_in_ms();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;

    while(1){
    printf("=============================== \n");
        ++count;
        {
#if (defined VANILLA)
            /* Image index */
            display_index = (buff_index + 1) %3;
            detect_index = (buff_index + 2) %3;
#elif (defined CONTENTION_FREE)
            display_index = (buff_index + 2) %3;
            detect_index = (buff_index) %3;
#else
    //fprintf(stderr, "ERROR: Set either ZERO_SLACK or CONTENTION_FREE or NEW_ZERO_SLACK in Makefile\n");
    //exit(0);
#endif
            const float nms = .45;    // 0.4F
            int local_nboxes = nboxes;
            detection *local_dets = dets;

            /* Fork fetch thread */
            if (!benchmark) custom_atomic_store_int(&run_fetch_in_thread, 1); // if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);

#if defined(VANILLA) 
            /* Fork Inference thread */
            custom_atomic_store_int(&run_detect_in_thread, 1); // if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
#endif

	    
	    /* display thread */
	    printf("start display");
            printf("\033[2J");
            printf("\033[1;1H");
            
            printf("\nB_fetch:%.1f\n", b_fetch);
            printf("\nTrafnsferdelay:%.1f\n", transfer_delay);
            printf("\nImage Waiting Time:%.1f", image_waiting_time);
            printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);
            printf("Objects:\n\n");

            double start_disp = get_time_in_ms();

            // nms = 0.45
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS){ // other dnn
                    do_nms_sort(local_dets, local_nboxes, classes, nms);
                }
                else { //yolo
                    diounms_sort(local_dets, local_nboxes, classes, nms, l.nms_kind, l.beta_nms);
                }
            }
            
            if (l.embedding_size) set_track_id(local_dets, local_nboxes, demo_thresh, l.sim_thresh, l.track_ciou_norm, l.track_history_size, l.dets_for_track, l.dets_for_show);
            
#ifdef V4L2

            if (!benchmark) {
                draw_detections_v3(frame[display_index].frame, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);

            }
            free_detections(local_dets, local_nboxes);

            draw_bbox_time = get_time_in_ms() - start_disp;

            /* Image display */
            display_in_thread(0);

#else
            /* original display thread */

            ++frame_id;
            if (demo_json_port > 0) {
                int timeout = 400000;
                printf("\n================\n");
                //send_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, demo_json_port, timeout);
                send_json(local_dets, local_nboxes, classes, demo_names, frame_id, demo_json_port, timeout);
            }

            printf("\n================\n");
            //char *http_post_server = "webhook.site/898bbd9b-0ddd-49cf-b81d-1f56be98d870";
            if (http_post_host && !send_http_post_once) {
                int timeout = 3;            // 3 seconds
                int http_post_port = 80;    // 443 https, 80 http
                if (send_http_post_request(http_post_host, http_post_port, filename,
                            local_dets, nboxes, classes, names, frame_id, ext_output, timeout))
                {
                    if (time_limit_sec > 0) send_http_post_once = 1;
                }
            }

            if (!benchmark) draw_detections_cv_v3(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);

            draw_bbox_time = get_time_in_ms() - start_disp;

            free_detections(local_dets, local_nboxes);

            //printf("\n================\n");
            if(!prefix){
                if (!dont_show) {
#ifdef DNN
                    show_image_mat(show_img, "Demo");
#else
                    show_image_mat(show_img, "Classifier Demo");
#endif                    
                    waitkey_start = get_time_in_ms();
                    //int c = wait_key_cv(1);
		            //cv::waitKey(1);
                    b_disp = get_time_in_ms() - waitkey_start;

                    if (c == 10) {
                        if (frame_skip == 0) frame_skip = 60;
                        else if (frame_skip == 4) frame_skip = 0;
                        else if (frame_skip == 60) frame_skip = 4;
                        else frame_skip = 0;
                    }
                    else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }

                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
                if(show_img) save_cv_jpg(show_img, buff);
            }

            // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
            if (mjpeg_port > 0 && show_img) {
                int port = mjpeg_port;
                int timeout = 400000;
                int jpeg_quality = 40;    // 1 - 100
                send_mjpeg(show_img, port, timeout, jpeg_quality);
            }

            // save video file
            if (output_video_writer && show_img) {
                write_frame_cv(output_video_writer, show_img);
                printf("\n cvWriteFrame \n");
            }

#endif
            /* display end */

            end_disp = get_time_in_ms();

            d_disp = end_disp - start_disp; 

#ifdef VANILLA

            /* Join Inference thread */
            pthread_join(inference_thread, 0);
#endif
            /* Join fetch thread */
            if (!benchmark) {
                pthread_join(fetch_thread, 0);
                free_image(det_s);
            }

#ifdef CONTENTION_FREE 
            /* Change infer image for next object detection cycle*/
            det_img = in_img;
            det_s = in_s;

            rtod_inference_thread(0);
#endif

            if (time_limit_sec > 0 && (get_time_point() - start_time_lim)/1000000 > time_limit_sec) {
                printf(" start_time_lim = %f, get_time_point() = %f, time spent = %f \n", start_time_lim, get_time_point(), get_time_point() - start_time_lim);
                break;
            }

            if (flag_exit == 1) break;

            if(delay == 0){
#ifndef V4L2
                if(!benchmark) release_mat(&show_img);
#endif
                show_img = det_img;
            }
#if defined(VANILLA)
            det_img = in_img;
            det_s = in_s;
#endif
            cycle_end = get_time_in_ms();
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;
            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            double after_1 = get_time_in_ms();    
            float curr = 1000000. / (after - before);
            float curr_1 = (after_1 - before_1);
            //fps = fps*0.9 + curr*0.1;  
            fps = 1000.0/curr_1;
            before = after;
            before_1 = after_1;

            float spent_time = (get_time_point() - start_time) / 1000000;
            frame_counter++;
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
        }
        cycle_array[cycle_index] = 1000./fps;
        cycle_index = (cycle_index + 1) % 4;
        slack_time = (MAX(d_infer, d_disp)) - (d_fetch);


#ifdef MEASUREMENT
        if (cnt >= CYCLE_OFFSET) push_data();

        /* Exit object detection cycle */
        if(cnt == ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET) - 1)) 
        {
            if(-1 == write_result(file_path))
            {
                /* return error */
                exit(0);
            }

            /* exit loop */
            break;
        }
#endif

        /* Increase count */
        if(cnt != ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET)-1)) cnt++;
        /* Change buffer index */
        buff_index = (buff_index + 1) % 3;
    }
    cnt = 0;

#ifdef MEASUREMENT
    /* Average data */
    printf("============ Darknet data ============\n");
    printf("Avg fetch execution time (ms) : %0.2f\n", e_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch blocking time (ms) : %0.2f\n", b_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch delay (ms) : %0.2f\n", d_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on cpu (ms) : %0.2f\n", e_infer_cpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on gpu (ms) : %0.2f\n", e_infer_gpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer delay (ms) : %0.2f\n", d_infer_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp execution time (ms) : %0.2f\n", e_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp blocking time (ms) : %0.2f\n", b_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp delay (ms) : %0.2f\n", d_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg slack (ms) : %0.2f\n", slack_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg E2E delay (ms) : %0.2f\n", e2e_delay_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg cycle time (ms) : %0.2f\n", cycle_time_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg transfer delay (ms) : %0.2f\n", transfer_delay_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg inter frame gap : %0.2f\n", inter_frame_gap_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg number of object : %0.2f\n", num_object_sum / OBJ_DET_CYCLE_IDX);
    printf("=====================================\n");
#endif

    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    // free memory
    free_image(in_s);
    free_detections(dets, nboxes);
    
#ifdef ASYNC
	//free buffer & cudaEvent
	free(hGlobal_layer_weights);
	cudaFree(global_layer_weights);
	cudaEventDestroy(copyEvent);
	for(int k = 0; k<net.n; k++) cudaEventDestroy(kernel[k]);
#endif

    free(avg);
    
    for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
    demo_index = (NFRAMES + demo_index - 1) % NFRAMES;
    for (j = 0; j < NFRAMES; ++j) {
        release_mat(&cv_images[j]);
    }
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    
    //int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}
 
#else
void rtod(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, 
        int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
        int benchmark, int benchmark_layers)
{
    fprintf(stderr, "R-TOD needs OpenCV for webcam images.\n");
}
#endif

