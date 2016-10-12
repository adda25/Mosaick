#ifndef _PHOTO_COMPOSER_
#define _PHOTO_COMPOSER_
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <thread>
#include <fstream>
#include <random>

#ifndef TTY_LOG
#define TTY_LOG 1
#endif

#ifndef USE_THREADS
#define USE_THREADS 0
#endif

class Mosaik
{
private:

  struct MosaikConf 
  {
    // Output size [Computed]
    uint width = 0;
    uint height = 0;
    // Pics size
    uint sub_pic_size = 50; // [Computed]
    uint divisions = 80;
    uint master_scale = 8;
    // Border
    bool border = true;
    uint border_size = 2; // px
    // Alpha blending
    bool alpha_blending = true;
    float alpha = 0.7;
    // Color transfer
    bool color_transfer = false;
    // Mult-offset 
    uint multi_offset = 3;
  } mosaik_conf;

  typedef struct 
  {
    cv::Mat image;         // Original resized
    cv::Mat loaded_image;  // Resized
    cv::Scalar mean;
    std::vector<std::pair<int,int>> at_col_row;
  } Pic;

  void _load_master_from_path(std::string path);
  void _pic_preproc(Pic &pic);
  void _alpha_blending(const cv::Mat src1, const cv::Mat src2, cv::Mat &dst, float alpha);
  void _similar_pic_for_img(cv::Mat target, Pic &out_similar_pic, const uint row, const uint col);
  uint _compares_rows_and_cols(Pic pic, const uint row, const uint col); 

  uint _state = 0;

  cv::Mat _master;
  std::vector<Pic> _pics;

public:
  cv::Mat output;

  Mosaik(const std::string master_path, const std::vector<std::string> folders) {
    load_master_and_pics(master_path, folders);
  };
  ~Mosaik() {};

  Mosaik load_master_and_pics(const std::string master_path, 
                              const std::vector<std::string> folders);
  Mosaik append_pics_for_folder(std::string folder);
  Mosaik create_output(); 

  Mosaik set_conf(uint divisions = 40,
                  bool border = true, 
                  uint border_size = 10, 
                  bool alpha_blending = true, 
                  float alpha = 0.8,
                  uint master_scale = 4); 
};

#endif // __cplusplus
#endif //_PHOTO_COMPOSER_