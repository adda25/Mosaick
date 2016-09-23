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
    uint divisions = 40;
    uint master_scale = 4;
    // Border
    bool border = true;
    uint border_size = 10; // px
    // Alpha blending
    bool alpha_blending = true;
    float alpha = 0.8;
    // Color transfer
    bool color_transfer = false;
  } mosaik_conf;

  typedef struct 
  {
    cv::Mat loaded_image; // Resized
    cv::Scalar mean;
    std::vector<std::pair<int,int>> at_col_row;
  } Pic;

  void _load_master_from_path(std::string path);
  void _pic_preproc(Pic &pic);
  void _alpha_blending(const cv::Mat src1, const cv::Mat src2, cv::Mat &dst, float alpha);
  void _similar_pic_for_img(cv::Mat target, Pic &out_similar_pic);

  uint _state = 0;

  cv::Mat _master;
  std::vector<Pic> _pics;

public:
  cv::Mat output;

  Mosaik() {};
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








class PhotoComposer
{
public:
  cv::Mat output;

  typedef struct 
  {
    cv::Mat image;
    cv::Mat resized_image;
    cv::Mat tr_image;
    cv::Scalar mean;
  } Pic;

  PhotoComposer() {};
  ~PhotoComposer() {};
 
  void load_pics_from_folder(std::string folder);
  void load_master_from_path(std::string path, int divisions);
  void create_output();

private:
  int _width  = 0;
  int _height = 0;
  int width  = 0;
  int height = 0;
  int _px_sub_img = 50;
  std::vector<Pic> _pics;
  std::vector<cv::Mat> _pics_hist;
  cv::Mat _master;

  void _tr_pic_to_square(Pic &p);
  void _alpha_blending(cv::Mat &sub_pic, cv::Rect rect_sub_pic);
  cv::Mat _get_most_similar_pic(cv::Rect rect);

  Pic _find_most_similar_image(const cv::Mat sub_master);
};

#endif // __cplusplus



#endif //_PHOTO_COMPOSER_