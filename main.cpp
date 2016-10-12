#include "photo_composer.hpp"

// export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

int 
main(int argc, char** argv) 
{
  std::vector<std::string> pics_paths;
  //pics_paths.push_back("/Users/adda/Pictures/*.jpg");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.JPG");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.jpg");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.jpeg");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.JPEG");
  std::string master_path = "/Users/adda/Desktop/foto/IMG-20160522-WA0008-R.jpg";
  Mosaik mosaik = Mosaik(master_path, pics_paths).create_output();
  cv::imwrite("./output.jpg", mosaik.output);
  return 0;
}

// Goods
// IMG-20160522-WA0008.jpg   m1 m111 -> d80
//IMG-20161009-WA0013.jpg   m2