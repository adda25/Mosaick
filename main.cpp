#include "photo_composer.hpp"
#include <iostream>

// export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

int 
main(int argc, char** argv) 
{
  std::vector<std::string> pics_paths;
  pics_paths.push_back("/Users/adda/Desktop/foto/*.JPG");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.jpg");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.jpeg");
  pics_paths.push_back("/Users/adda/Desktop/foto/*.JPEG");
  std::string master_path = "/Users/adda/Desktop/foto/IMG-20160522-WA0008-R.jpg";
  Mosaik mosaik = Mosaik(master_path, pics_paths);

  std::string cmd = "";
  float val = 0;
  while (true) {
  	std::cout << "Cmd: ";
  	std::cin >> cmd;
  	if (cmd == "exc") {
  		mosaik.create_output();
  		cv::imwrite("./output.jpg", mosaik.output);
  		cmd = "";
  		continue;
  	} 
  	if (cmd == "exit") {
  		break;
  	} 
  	std::cout << "Val: ";
  	std::cin >> val;
  	if (cmd == "div") {
  		mosaik.mosaik_conf.divisions = (int)val;
  		mosaik._load_master_from_path(master_path);
  	} else if (cmd == "alpha") {
  		mosaik.mosaik_conf.alpha = val;
  	} else if (cmd == "offset") {
  		mosaik.mosaik_conf.multi_offset = (uint)val;
  	} else if (cmd == "border") {
  		mosaik.mosaik_conf.border_size = (uint)val;
  	} else if (cmd == "scale") {
  		mosaik.mosaik_conf.master_scale = (uint)val;
  		mosaik._load_master_from_path(master_path);
  	}
  }

  return 0;
}

// Goods
// IMG-20160522-WA0008.jpg   m1 m111 -> d80
//IMG-20161009-WA0013.jpg   m2