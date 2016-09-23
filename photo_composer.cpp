#include "photo_composer.hpp"

std::random_device _seed;
std::mt19937 rng(_seed());

Mosaik 
Mosaik::load_master_and_pics(const std::string master_path, const std::vector<std::string> folders)
{
  _pics.clear();
  _load_master_from_path(master_path);
  for (auto &f : folders) {
    append_pics_for_folder(f);
  }
  _state = 2;
  return *this;
}

Mosaik 
Mosaik::append_pics_for_folder(std::string folder)
{
  if (_state != 1) { return *this; }
  std::vector<cv::String> fn; 
  cv::glob(folder, fn, true); 
  for (auto &image_name : fn) {
    Pic pic;
    pic.loaded_image = cv::imread(image_name);
    _pic_preproc(pic); 
    cv::resize(pic.loaded_image, pic.loaded_image, 
               cv::Size(mosaik_conf.sub_pic_size, mosaik_conf.sub_pic_size));
    pic.mean = cv::mean(pic.loaded_image);
    _pics.push_back(pic);
    #if TTY_LOG
    std::cout << "Loading image: " << std::addressof(image_name) - std::addressof(fn[0]) << " / " << fn.size() <<  " " << image_name << std::endl;   
    #endif
  }
  return *this;
}

void 
Mosaik::_load_master_from_path(std::string path)
{
  int _width  = 0;
  int _height = 0;
  _master = cv::imread(path);
  cv::resize(_master, _master, cv::Size(_master.cols * mosaik_conf.master_scale, 
                                        _master.rows * mosaik_conf.master_scale));
  _width  = _master.cols; 
  _height = _master.rows;
  mosaik_conf.sub_pic_size = (uint)((float)_width / (float)mosaik_conf.divisions);
  //std::cout << "Output width: " << _width << " height: " << _height << "  --> " << _px_sub_img * mosaik_conf.divisions  << std::endl; 
  // Set width
  if (mosaik_conf.sub_pic_size * mosaik_conf.divisions < _width) {
    mosaik_conf.width = _width - mosaik_conf.sub_pic_size;
  } else {
    mosaik_conf.width = _width;
  }
  // Set height
  if (_height / mosaik_conf.sub_pic_size < mosaik_conf.divisions) {
    mosaik_conf.height = _height - mosaik_conf.sub_pic_size;
  } else {
    mosaik_conf.height = _height;
  }
  _state = 1;
}

Mosaik 
Mosaik::create_output()
{
  if (_state != 2) { return *this; }
  cv::Mat _output = cv::Mat(mosaik_conf.height, mosaik_conf.width, CV_8UC3);
  for (int i = 0; i < mosaik_conf.width / mosaik_conf.sub_pic_size; i++) {
    for (int k = 0; k < mosaik_conf.height / mosaik_conf.sub_pic_size; k++) {
      //cv::Rect sub_mst_border = cv::Rect(i * _px_sub_img+2, k * _px_sub_img+2, _px_sub_img-4, _px_sub_img-4);
      cv::Rect sub_mst_rect = cv::Rect(i * mosaik_conf.sub_pic_size, 
                                       k * mosaik_conf.sub_pic_size, 
                                       mosaik_conf.sub_pic_size, 
                                       mosaik_conf.sub_pic_size);
      cv::Mat sub_master = _master(sub_mst_rect);
      Pic most_sim_im; 
      _similar_pic_for_img(sub_master, most_sim_im);
      cv::Mat out_sub_mat;
      // Do alpha blending if requested
      if (mosaik_conf.alpha_blending) {
        _alpha_blending(most_sim_im.loaded_image, sub_master, out_sub_mat, mosaik_conf.alpha);
      } else {
        most_sim_im.loaded_image.copyTo(out_sub_mat);
      }
      // Create border if requested
      if (mosaik_conf.border) {
        cv::Mat tr_image_no_border;
        cv::Mat tr_image_no_border2;
        out_sub_mat.copyTo(tr_image_no_border);
        cv::Mat tr_image_border(tr_image_no_border.rows, tr_image_no_border.cols, tr_image_no_border.depth());
        cv::resize(tr_image_no_border, tr_image_no_border2, cv::Size(tr_image_no_border.cols - 2 * mosaik_conf.border_size, tr_image_no_border.rows - 2 * mosaik_conf.border_size));
        cv::copyMakeBorder(tr_image_no_border2, tr_image_border, mosaik_conf.border_size, mosaik_conf.border_size, mosaik_conf.border_size, mosaik_conf.border_size, cv::BORDER_CONSTANT, cv::Scalar(255,255,255,255));
        tr_image_border.copyTo(_output(sub_mst_rect));
      } else {     
        out_sub_mat.copyTo(_output(sub_mst_rect));
      }
    }
  }
  #if TTY_LOG
  std::cout << "Finish to create image" << std::endl;
  #endif
  //transfer_image(_master, _output, _output);
  _output.copyTo(output);
  _state = 0;
  return *this;
}


Mosaik 
Mosaik::set_conf(uint divisions,
                 bool border, 
                 uint border_size, 
                 bool alpha_blending, 
                 float alpha,
                 uint master_scale)
{
  mosaik_conf.divisions = divisions;
  mosaik_conf.border = border;
  mosaik_conf.border_size = border_size;
  mosaik_conf.alpha_blending = alpha_blending;
  mosaik_conf.master_scale = master_scale;
  return *this;
}

void 
Mosaik::_pic_preproc(Pic &pic)
{
  if (pic.loaded_image.rows == pic.loaded_image.cols) return;
  int square_size = pic.loaded_image.rows > pic.loaded_image.cols ? pic.loaded_image.cols : pic.loaded_image.rows;
  cv::Mat squared_image = pic.loaded_image(cv::Rect(0, 0, square_size, square_size));
  pic.loaded_image = squared_image;
}


void 
Mosaik::_alpha_blending(const cv::Mat src1, const cv::Mat src2, cv::Mat &dst, float alpha)
{
  float _alpha = alpha > 1.0 ? 1.0 : alpha;
  _alpha = alpha < 0.0 ? 0.0 : alpha;  
  cv::addWeighted(src1, _alpha, src2, 1 - _alpha, 0.0, dst);
}


void 
Mosaik::_similar_pic_for_img(cv::Mat target, Pic &out_similar_pic)
{
  cv::Scalar m_mean = cv::mean(target);
  int min_dist = 1000000000;
  int min_dist_index = 0;
  for (auto &p : _pics) {
    int dist = fabs(p.mean[0] - m_mean[0]) + fabs(p.mean[1] - m_mean[1]) + fabs(p.mean[2] - m_mean[2]); 
    if (dist < min_dist) {
      min_dist = dist;
      min_dist_index = std::addressof(p) - std::addressof(_pics[0]);
    }
  }
  std::vector<int> good_indexs;
  for (auto &p : _pics) {
    int dist = fabs(p.mean[0] - m_mean[0]) + fabs(p.mean[1] - m_mean[1]) + fabs(p.mean[2] - m_mean[2]); 
    if (dist < min_dist * 2.0) {
      int good_index = std::addressof(p) - std::addressof(_pics[0]);
      good_indexs.push_back(good_index);
    }
  }
  if (good_indexs.size() > 1) {
    std::uniform_int_distribution<int> gen(0, good_indexs.size()-1); // uniform, unbiased
    int r = gen(rng);
    min_dist_index = good_indexs[r];
  } 
  out_similar_pic = _pics[min_dist_index];
}








void transfer_image(cv::Mat src, cv::Mat tar, cv::Mat &result);

void 
PhotoComposer::load_pics_from_folder(std::string folder)
{
  std::vector<cv::String> fn;	
  cv::glob(folder, fn, true); // recurse
  std::vector<std::thread> threads(fn.size());
  int i = 0;
  for (auto &image_name : fn) {
    Pic pic;
    pic.image = cv::imread(image_name);
    _tr_pic_to_square(pic);
    cv::resize(pic.image, pic.resized_image, cv::Size(_px_sub_img, _px_sub_img));
    pic.mean = cv::mean(pic.resized_image);
    pic.image = cv::Mat();
  	_pics.push_back(pic);
  	//threads.at(i) = std::thread (_load_image, image_name, _pics.back());  
    //printf("Loading image: %s \r", image_name.c_str());
  	std::cout << "Loading image: " << i << " / " << fn.size() <<  " " << image_name << std::endl;   
    i++;
  }
  /*for (int k = 0; k < i; k++) {
  	threads.at(k).join();
  	std::cout << "Loading pics: " << k + 1 << " / " << fn.size() << std::endl;
  }*/
  //_tr_pics_to_square(_pics);
}

void 
PhotoComposer::load_master_from_path(std::string path, int divisions)
{
  _master = cv::imread(path);
  cv::resize(_master, _master, cv::Size(_master.cols * 6, _master.rows * 6));
  _width  = _master.cols; 
  _height = _master.rows;

  _px_sub_img = (int)_width / divisions;
  std::cout << "Output width: " << _width << " height: " << _height << "  --> " << _px_sub_img * divisions  << std::endl; 
  // Set width
  if (_px_sub_img * divisions < _width) {
    width = _width - _px_sub_img;
  } else {
    width = _width;
  }
  // Set height
  if (_height / _px_sub_img < divisions) {
    height = _height - _px_sub_img;
  } else {
    height = _height;
  }
}


void 
PhotoComposer::create_output()
{
  cv::Mat _output = cv::Mat(height, width, CV_8UC3);
  for (int i = 0; i < width / _px_sub_img; i++) {
    for (int k = 0; k < height / _px_sub_img; k++) {
      std::cout << "Image: " << i + k << std::endl;
      //cv::Rect sub_mst_border = cv::Rect(i * _px_sub_img+2, k * _px_sub_img+2, _px_sub_img-4, _px_sub_img-4);
      cv::Rect sub_mst_rect = cv::Rect(i * _px_sub_img, k * _px_sub_img, _px_sub_img, _px_sub_img);
      cv::Mat sub_master = _master(sub_mst_rect);
      Pic most_sim_im = _find_most_similar_image(sub_master);
      most_sim_im.resized_image.copyTo(most_sim_im.tr_image);
      _alpha_blending(most_sim_im.tr_image, sub_mst_rect);

      cv::Mat tr_image_no_border;
      cv::Mat tr_image_no_border2;
      most_sim_im.tr_image.copyTo(tr_image_no_border);
      cv::Mat tr_image_border(tr_image_no_border.rows, tr_image_no_border.cols, tr_image_no_border.depth());
      cv::resize(tr_image_no_border, tr_image_no_border2, cv::Size(tr_image_no_border.cols - 20, tr_image_no_border.rows - 20));
      cv::copyMakeBorder(tr_image_no_border2, tr_image_border, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(255,255,255,255));
      tr_image_border.copyTo(_output(sub_mst_rect));
    }
  }
  std::cout << "Finish to create image" << std::endl;
  //transfer_image(_master, _output, _output);
  _output.copyTo(output);
}

void 
PhotoComposer::_alpha_blending(cv::Mat &sub_pic, cv::Rect rect_sub_pic)
{
  cv::Mat sub_master = _master(rect_sub_pic); 
  cv::Mat out;
  cv::Mat out2;
  cv::Mat entry, target_colorspace;
  sub_master.copyTo(target_colorspace);
  //sub_pic.copyTo(entry);
  cv::Scalar master_mean = cv::mean(sub_master);
  sub_pic.copyTo(entry);
  target_colorspace.setTo(master_mean);  //Scalar.val[0-2] used 
  cv::addWeighted(sub_pic, 0.8, target_colorspace, 0.2, 0.0, out);
  //transfer_image(target_colorspace, entry, out2);
  out.copyTo(sub_pic);
}



PhotoComposer::Pic 
PhotoComposer::_find_most_similar_image(const cv::Mat sub_master)
{
  cv::Scalar m_mean = cv::mean(sub_master);
  int min_dist = 1000000000;
  int min_dist_index = 0;
  for (auto &p : _pics) {
    int dist = fabs(p.mean[0] - m_mean[0]) + fabs(p.mean[1] - m_mean[1]) + fabs(p.mean[2] - m_mean[2]); 
    if (dist < min_dist) {
      min_dist = dist;
      min_dist_index = std::addressof(p) - std::addressof(_pics[0]);
    }
  }
  std::vector<int> good_indexs;
  for (auto &p : _pics) {
    int dist = fabs(p.mean[0] - m_mean[0]) + fabs(p.mean[1] - m_mean[1]) + fabs(p.mean[2] - m_mean[2]); 
    if (dist < min_dist * 2.0) {
      int good_index = std::addressof(p) - std::addressof(_pics[0]);
      good_indexs.push_back(good_index);
    }
  }
  if (good_indexs.size() > 1) {
    std::uniform_int_distribution<int> gen(0, good_indexs.size()-1); // uniform, unbiased
    int r = gen(rng);
    std::cout << "Size: " << good_indexs.size() << " " << r << std::endl;
    min_dist_index = good_indexs[r];
  } 
  return _pics[min_dist_index];
}


void 
PhotoComposer::_tr_pic_to_square(Pic &p)
{
  if (p.image.rows == p.image.cols) return;
  int square_size = p.image.rows > p.image.cols ? p.image.cols : p.image.rows;
  cv::Mat squared_image = p.image(cv::Rect(0, 0, square_size, square_size));
  p.image = squared_image;
}





using namespace cv;
using namespace std;


class imageInfo{
public:
    double lMean, lStd, aMean, aStd, bMean, bStd;
};

void image_stats(Mat img, imageInfo *info);

void transfer_image(cv::Mat src, cv::Mat tar, cv::Mat &result)
{
  vector<Mat> mv;
  imageInfo srcInfo, tarInfo;
  cvtColor(src, src, CV_BGR2Lab);
  cvtColor(tar, tar, CV_BGR2Lab);
  image_stats(src, &srcInfo);
  image_stats(tar, &tarInfo);
  split(tar, mv);
  Mat l = mv[0];
  Mat a = mv[1];
  Mat b = mv[2];
  /*pixel color modify*/
  for (int i = 0; i<l.rows; i++){
      for (int j = 0; j<l.cols; j++){
          double li = l.data[l.step[0] * i + l.step[1] * j];
          if (i == 426 && j == 467)
          li -= tarInfo.lMean;
          li = (tarInfo.lStd / srcInfo.lStd)*li;
          li += srcInfo.lMean;
          if (li > 255) li = 255;
          if (li < 0) li = 0;
          l.data[l.step[0] * i + l.step[1] * j] = li;
      }
  }
  for (int i = 0; i<a.rows; i++){
      for (int j = 0; j<a.cols; j++){
          double ai = a.data[a.step[0] * i + a.step[1] * j];
          ai -= tarInfo.aMean;
          ai = (tarInfo.aStd / srcInfo.aStd)*ai;
          ai += srcInfo.aMean;
          if (ai > 255) ai = 255;
          if (ai < 0) ai = 0;
          a.data[a.step[0] * i + a.step[1] * j] = ai;
      }
  }
  for (int i = 0; i<b.rows; i++){
      for (int j = 0; j<b.cols; j++){
          double bi = b.data[b.step[0] * i + b.step[1] * j];
          bi -= tarInfo.bMean;
          bi = (tarInfo.bStd / srcInfo.bStd)*bi;
          bi += srcInfo.bMean;
          if (bi > 255) bi = 255;
          if (bi < 0) bi = 0;
          b.data[b.step[0] * i + b.step[1] * j] = bi;
      }
  }
  mv.clear();
  mv.push_back(l);
  mv.push_back(a);
  mv.push_back(b);
  merge(mv, result);
  cvtColor(result, result, CV_Lab2BGR);
}

void image_stats(Mat img, imageInfo *info){
    int Max=0;
    vector<Mat> mv;
    vector<int> vl, va, vb;
    split(img, mv);
    Mat l = mv[0];
    Mat a = mv[1];
    Mat b = mv[2];

    /*statistics L space*/
    for (int i = 0; i<l.rows; i++){
        for (int j = 0; j<l.cols; j++){
            int li = l.data[l.step[0] * i + l.step[1] * j];
            vl.push_back(li);
        }
    }
    double sum_l = std::accumulate(vl.begin(), vl.end(), 0.0);
    double mean_l = sum_l / vl.size();
    std::vector<double> diff_l(vl.size());
    std::transform(vl.begin(), vl.end(), diff_l.begin(),
        std::bind2nd(std::minus<double>(), mean_l));
    double sq_sum_l = std::inner_product(diff_l.begin(), diff_l.end(), diff_l.begin(), 0.0);
    double stdev_l = std::sqrt(sq_sum_l / vl.size());
    info->lMean = mean_l;
    info->lStd = stdev_l;

    /*statistics A space*/
    for (int i = 0; i<a.rows; i++){
        for (int j = 0; j<a.cols; j++){
            int ai = a.data[a.step[0] * i + a.step[1] * j];
            va.push_back(ai);
        }
    }
    double sum_a = std::accumulate(va.begin(), va.end(), 0.0);
    double mean_a = sum_a / va.size();
    std::vector<double> diff_a(va.size());
    std::transform(va.begin(), va.end(), diff_a.begin(),
        std::bind2nd(std::minus<double>(), mean_a));
    double sq_sum_a = std::inner_product(diff_a.begin(), diff_a.end(), diff_a.begin(), 0.0);
    double stdev_a = std::sqrt(sq_sum_a / va.size());
    info->aMean = mean_a;
    info->aStd = stdev_a;

    /*statistics B space*/
    for (int i = 0; i<b.rows; i++){
        for (int j = 0; j<b.cols; j++){
            int bi = b.data[b.step[0] * i + b.step[1] * j];
            vb.push_back(bi);
        }
    }
    double sum_b = std::accumulate(vb.begin(), vb.end(), 0.0);
    double mean_b = sum_b / vb.size();
    std::vector<double> diff_b(vb.size());
    std::transform(vb.begin(), vb.end(), diff_b.begin(),
        std::bind2nd(std::minus<double>(), mean_b));
    double sq_sum_b = std::inner_product(diff_b.begin(), diff_b.end(), diff_b.begin(), 0.0);
    double stdev_b = std::sqrt(sq_sum_b / vb.size());
    info->bMean = mean_b;
    info->bStd = stdev_b;


}




/*
cv::Mat
PhotoComposer::_get_most_similar_pic(cv::Rect rect)
{
  cv::Mat sub_master = _master(rect);
  cv::Mat hist_master = _calc_histogram(sub_master);
  double min_err = 100000000;
  int index = 0;
  for (int i = 0; i < _pics_hist.size(); i++) {
    std::cout << i << std::endl;
    double err = _compare_histogram(_pics_hist[i], hist_master);
    if (err < min_err) {
      min_err = err;
      index = i;
    }
  }
  return _pics[index];
}

cv::Mat 
PhotoComposer::_calc_histogram(cv::Mat pic) 
{
  cv::Mat hist;
  int imgCount = 1;
  int dims = 3;
  const int sizes[] = {256, 256, 256};
  const int channels[] = {0,1,2};
  float rRange[] = {0,256};
  float gRange[] = {0,256};
  float bRange[] = {0,256};
  const float *ranges[] = {rRange, gRange, bRange};
  cv::Mat mask = cv::Mat();
  cv::calcHist(&pic, imgCount, channels, mask, hist, dims, sizes, ranges);
  return hist;
}

double
PhotoComposer::_compare_histogram(cv::Mat h1, cv::Mat h2)
{
  return cv::compareHist(h1, h2, CV_COMP_CORREL);
}

  //std::mt19937 rng;
  //rng.seed(std::random_device()());
  //std::uniform_int_distribution<std::mt19937::result_type> dist6(0, 12 - 1);
  //cv::applyColorMap(sub_pic, sub_pic, dist6(rng));

*/




