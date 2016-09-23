#include "photo_composer.hpp"

// try HSV
// new logic similar pics
// user inteface
// - alpha/beta
// - border
// - divisions
// - color transfer

// export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

int 
main(int argc, char** argv) 
{
  std::vector<std::string> pics_paths;
  pics_paths.push_back("/Users/adda/Pictures/SDCel/*.jpg");
  pics_paths.push_back("/Users/adda/Pictures/SDCel/*.JPG");
  std::string master_path = "./master/gatto.jpg";
  Mosaik mosaik = Mosaik().load_master_and_pics(master_path, pics_paths).create_output();
  cv::imwrite("./output.jpg", mosaik.output);


  //mosaik.set_conf(600, false).load_master_and_pics(master_path, pics_paths).create_output();
  //cv::imwrite("./output1.jpg", mosaik.output);

  return 0;
}

/*int 
main(int argc, char** argv) 
{
  std::string pics_path = "/Users/adda/Pictures/*.jpg";
  std::string master_path = "./master/gatto.jpg";
  PhotoComposer pc = PhotoComposer();
  pc.load_master_from_path(master_path, 40);
  pc.load_pics_from_folder(pics_path);
  pc.create_output();
  //cv::imshow("output", pc.output);
  //cv::waitKey(50000);
  cv::imwrite("./output.jpg", pc.output);
  return 0;
}*/