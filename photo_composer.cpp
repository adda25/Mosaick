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
  return *this;
}

Mosaik 
Mosaik::append_pics_for_folder(std::string folder)
{
  std::vector<cv::String> fn; 
  cv::glob(folder, fn, true); 
  for (auto &image_name : fn) {
    Pic pic;
    pic.image = cv::imread(image_name);
    _pic_preproc(pic); 
    cv::resize(pic.image, pic.image, 
               cv::Size(mosaik_conf.max_load_size, mosaik_conf.max_load_size));
    pic.mean = cv::mean(pic.image);
    pic.image.copyTo(pic.loaded_image);
    _pics.push_back(pic);
    #if TTY_LOG
    std::cout << "Loading image: " << (std::addressof(image_name) - std::addressof(fn[0]) + 1) << " / " << fn.size() <<  " " << image_name << std::endl;   
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
  std::cout << "SUB SIZE: " << mosaik_conf.sub_pic_size << std::endl;
}

Mosaik 
Mosaik::create_output()
{
  for (auto &p : _pics) {
    cv::resize(p.image, p.loaded_image, 
               cv::Size(mosaik_conf.sub_pic_size, mosaik_conf.sub_pic_size));
  }
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
      _similar_pic_for_img(sub_master, most_sim_im, i, k);
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
  if (pic.image.rows == pic.image.cols) return;
  int square_size = pic.image.rows > pic.image.cols ? pic.image.cols : pic.image.rows;
  cv::Mat squared_image = pic.image(cv::Rect(0, 0, square_size, square_size));
  pic.image = squared_image;
}


void 
Mosaik::_alpha_blending(const cv::Mat src1, const cv::Mat src2, cv::Mat &dst, float alpha)
{
  float _alpha = alpha > 1.0 ? 1.0 : alpha;
  _alpha = alpha < 0.0 ? 0.0 : alpha;  
  cv::addWeighted(src1, _alpha, src2, 1 - _alpha, 0.0, dst);
}


void 
Mosaik::_similar_pic_for_img(cv::Mat target, Pic &out_similar_pic, const uint row, const uint col)
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
    if (dist < min_dist * mosaik_conf.multi_offset) {
      int good_index = std::addressof(p) - std::addressof(_pics[0]);
      good_indexs.push_back(good_index);
    }
  }
  if (good_indexs.size() > 1) {
    /*uint min_dist = 0;
    uint min_idx = 0;
    for (auto &i : good_indexs) {
      int d =  _compares_rows_and_cols(_pics[i], row, col);
      if (d > min_dist) {
        min_dist = d;
        min_idx = i;
      }
    }
    min_dist_index = min_idx; 
    _pics[min_dist_index].at_col_row.push_back(std::make_pair(row, col)); */
    std::uniform_int_distribution<int> gen(0, good_indexs.size()-1); // uniform, unbiased
    int r = gen(rng);
    min_dist_index = good_indexs[r];
  } 
  out_similar_pic = _pics[min_dist_index];
}


uint 
Mosaik::_compares_rows_and_cols(Pic pic, const uint row, const uint col)
{
  int distance = 10000000;
  for (auto &cr : pic.at_col_row) {
    distance  = ((cr.first + cr.second) - (row + col)) < distance ? distance : ((cr.first + cr.second) - (row + col));
  }
  return std::abs(distance);
}




