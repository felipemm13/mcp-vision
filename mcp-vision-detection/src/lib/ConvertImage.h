#include <opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgcodecs/legacy/constants_c.h>

#include <vector>
#include <string>

using namespace std;
using namespace cv;

#ifndef CONVERTIMAGE_H_
#define CONVERTIMAGE_H_

/**
 * Classe que converte as imagens para base64 e virse e versa
*/
class ImagemConverter {
public:
	ImagemConverter();
    virtual ~ImagemConverter();
	
	cv::Mat str2mat(const string& imageBase64);
	string mat2str(const Mat& img);

private:
    static const std::string base64_chars;
    static inline bool is_base64(unsigned char c);
    std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
    std::string base64_decode(const std::string& encoded_string);
};

#endif /* CONVERTIMAGE_H_ */
