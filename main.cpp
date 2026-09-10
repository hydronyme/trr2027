#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <vpi/Array.h>
#include <vpi/CUDAInterop.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/BoxFilter.h>
#include <vpi/algo/CannyEdges.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/EqualizeHist.h>

#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cuda_runtime.h>
#include <nvbufsurface.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring> // for memset
#include <format>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono> // for sleep
#include <thread> // for sleep


#include "line_analysis.cuh"

#define MAX_BORDERS 512

// ---------------------------------------------------------------------
// Petit helper pour transformer un code d'erreur VPI en exception C++.
// ---------------------------------------------------------------------

#define CHECK_VPI(STMT)                                                        \
  do {                                                                         \
    VPIStatus status = (STMT);                                                 \
    if (status != VPI_SUCCESS) {                                               \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];                              \
      vpiGetLastStatusMessage(buffer, sizeof(buffer));                         \
      std::ostringstream oss;                                                  \
      oss << "Erreur VPI (" << vpiStatusGetName(status) << "): " << buffer;    \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define CHECK_CUDA_STATUS(STMT)                                                \
  do {                                                                         \
    cudaError_t status = (STMT);                                               \
    if (status != cudaSuccess) {                                               \
      std::ostringstream ss;                                                   \
      ss << cudaGetErrorString(status);                                        \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0);

class CameraAnalysis {
public:
  CameraAnalysis()
      : param_canny_min_(150), param_canny_max_(200), width(1280), height(720),
        vh_(200), u0_(640), beta_x_(0.13646387832699614),
        beta_y_(67.19999999999997) {
    initVPI();
  }

  ~CameraAnalysis() {
    if (stream) vpiStreamDestroy(stream);
          gst_element_set_state(pipeline, GST_STATE_NULL);
          gst_object_unref(pipeline);
    // if (payload_eq) vpiStreamDestroy(payload_eq);
    // if (payload_canny) vpiStreamDestroy(payload_canny);
  }

  void loop() {
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    while (true) { }
  }

  int64 timer_0;
  int64 timer_1;
  int64 timer_2;
  int64 timer_3;
  int64 timer_4;
  int64 timer_5;
  int64 timer_6 = 0;

private:
 int init_gstreamer() {

          // const std::string filename = "./road/cam_2974997494746.jpg";
          //const std::string filename = "../../road/1000032674.jpg";
          const std::string filename = "../../road/image_1280x720.jpg";
          gst_init(NULL, NULL);

         std::string pipeline_video_str = 
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)1280, height=(int)720, "
            "framerate=(fraction)60/1, format=(string)NV12 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, "
            "width=(int)1280, height=(int)720, "
            "format=(string)NV12 ! "
            "appsink name=mysink emit-signals=true "
            "max-buffers=1 drop=true sync=false";

         std::string pipeline_file_str =
            "filesrc location=" + filename + " ! " 
            "jpegdec ! "
            "imagefreeze ! "
            "videoconvert ! "
            "video/x-raw,format=NV12,framerate=60/1 ! "
            "appsink name=mysink emit-signals=true max-buffers=1 drop=true sync=true";

          GError *error = nullptr;
          pipeline = gst_parse_launch(pipeline_video_str.c_str(), nullptr);
          //pipeline = gst_parse_launch(pipeline_file_str.c_str(), nullptr);
          if (!pipeline) {
            std::cerr << "Impossible de créer le pipeline" << std::endl;
            if (error) {
                std::cerr << "GStreamer : "
                          << error->message
                          << std::endl;
                g_error_free(error);
            }
            return -1;
          }
          appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
          if (!appsink) {
            std::cerr << "appsink introuvable" << std::endl;
            gst_object_unref(pipeline);
            return -1;
        }
          g_signal_connect( appsink, "new-sample", G_CALLBACK(CameraAnalysis::onNewSample), this);
          return 0;
  }

  void initVPI() {
    init_gstreamer();

    CHECK_CUDA_STATUS(cudaStreamCreate(&cudaStream));
    CHECK_VPI(vpiStreamCreateWrapperCUDA(cudaStream, 0, &stream));
    vpiCreateEqualizeHist(VPI_BACKEND_CUDA, VPI_IMAGE_FORMAT_U8, &payload_eq);
    vpiCreateCannyEdgeDetector(VPI_BACKEND_CUDA, width, height, &payload_canny);
    vpiInitCannyEdgeDetectorParams(&canny_params);

    // Allocate pointers to be wrapped and create wrapped images
    vpiInitImageWrapperParams(&params);
    wrapParams.colorSpec = VPI_COLOR_SPEC_DEFAULT;

    cudaMalloc(&d_result, height * sizeof(int)); // Alloue sur le GPU

    // Optionnel : initialise d_result avec des zéros (si nécessaire)
    cudaMemcpy(d_result, h_init, height * sizeof(int), cudaMemcpyHostToDevice);
    // delete[] h_init; // Libère l'initialisation temporaire

    // Allocate CUDA pitch
    CHECK_CUDA_STATUS(cudaMalloc(&cudaGray, width * height * sizeof(uint8_t)));
    CHECK_CUDA_STATUS(cudaMalloc(&cudaBlur, width * height * sizeof(uint8_t)));
    CHECK_CUDA_STATUS(cudaMalloc(&cudaEq, width * height * sizeof(uint8_t)));
    CHECK_CUDA_STATUS(cudaMalloc(&cudaCanny, width * height * sizeof(uint8_t)));

    // Alloue avec pitch
    /*
    size_t pitch;
    CHECK_CUDA_STATUS(cudaMallocPitch(&cudaGray, &pitch, width * sizeof(uint8_t), height)); 
    CHECK_CUDA_STATUS(cudaMallocPitch(&cudaBlur, &pitch, width * sizeof(uint8_t), height));
    CHECK_CUDA_STATUS(cudaMallocPitch(&cudaEq, &pitch, width * sizeof(uint8_t), height)); 
    CHECK_CUDA_STATUS(cudaMallocPitch(&cudaCanny, &pitch, width * sizeof(uint8_t), height));
    */

    // Wrap CUDA pitches directly in VPIImage
    VPIImageData dataGray = GetGenericPLData( width, height, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaGray);
    VPIImageData dataBlur = GetGenericPLData( width, height, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaBlur);
    VPIImageData dataEq = GetGenericPLData( width, height, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaEq);
    VPIImageData dataCanny = GetGenericPLData( width, height, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaCanny);

    // 1. Allocation d'une image VPI forcée en mémoire Device CUDA
    CHECK_VPI(vpiImageCreateWrapper(&dataGray, &wrapParams, VPI_BACKEND_CUDA, &vpiGray));
    CHECK_VPI(vpiImageCreateWrapper(&dataBlur, &wrapParams, VPI_BACKEND_CUDA, &vpiBlur));
    CHECK_VPI( vpiImageCreateWrapper(&dataEq, &wrapParams, VPI_BACKEND_CUDA, &vpiEq));
    CHECK_VPI(vpiImageCreateWrapper(&dataCanny, &wrapParams, VPI_BACKEND_CUDA, &vpiCanny));
  }

  VPIImageData GetGenericPLData(int width, int height, VPIImageBufferType bufferType, VPIByte *pBase) {
    VPIImageBufferPitchLinear dataInterne = {};
    dataInterne.format = VPI_IMAGE_FORMAT_U8;
    dataInterne.numPlanes = 1;
    dataInterne.planes[0].width = width;
    dataInterne.planes[0].height = height;
    dataInterne.planes[0].pitchBytes = sizeof(uint8_t) * width;
    dataInterne.planes[0].pixelType = VPI_PIXEL_TYPE_INVALID;
    dataInterne.planes[0].offsetBytes = 0;
    dataInterne.planes[0].pBase = pBase;
    VPIImageData vpiImgData = {};
    vpiImgData.bufferType = bufferType;
    vpiImgData.buffer.pitch = dataInterne;
    return vpiImgData;
  }

  struct TargetResult {
    int u = -1;
    int v = 0;
    double x = 0.0;
    double y = 0.0;
    bool found = false;
  };

  void writeCudaImg(uint8_t *cudaData,std::string file) {
    cv::Mat img(height, width, CV_8UC1);
    std::cout << "toto" << '\n';
    CHECK_CUDA_STATUS(cudaMemcpy(img.data, cudaData, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost)); 
    cv::imwrite(file, img);
  }

  static GstFlowReturn onNewSample(GstAppSink* appsink, gpointer user_data);

  bool createVPIImageDataFromGstBuffer(GstBuffer *gstBuffer,GstCaps *caps) {
    timer_1= cv::getTickCount();
    nb_loop++;
    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(gstBuffer, &map, GST_MAP_READ)) return false;
    timer_2= cv::getTickCount();
    dataGray = GetGenericPLData(width,height,VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, (VPIByte*) map.data) ;
    CHECK_VPI(vpiImageCreateWrapper(&dataGray, &wrapParams, VPI_BACKEND_CUDA, &vpiGray));
    CHECK_VPI(vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, vpiGray, vpiBlur, 5, 5, VPI_BORDER_ZERO));
    CHECK_VPI(vpiSubmitEqualizeHist(stream, VPI_BACKEND_CUDA, payload_eq, vpiBlur, vpiEq));
    CHECK_VPI(vpiSubmitCannyEdgeDetector(stream, VPI_BACKEND_CUDA, payload_canny, vpiEq, vpiCanny, param_canny_max_, param_canny_min_, 255, 0, &canny_params));
    timer_3= cv::getTickCount();
    submitLineAnalysis(cudaEq, cudaCanny, width, height, cudaStream, d_result);
    CHECK_VPI(vpiStreamSync(stream));
    cudaMemcpy(h_result, d_result, height * sizeof(int), cudaMemcpyDeviceToHost);
    timer_4= cv::getTickCount();

    int count=0;
    for (int i = 0; i < 720; i++) {
      if(h_result[i]!=-1) 
      {
        count++;
      }
    }
    std::cout << "R:" << count << " points" << '\n';
    //std::cout << std::endl;
    if (nb_loop==22) {
        writeCudaImg(cudaCanny,"canny.png"); 
        writeCudaImg(cudaEq,"eq.png"); 

        cv::Mat imgGray( height, width, CV_8UC1, map.data, width);
        cv::imwrite("gray.png", imgGray);

        cv::Mat img;
        cv::Mat nv12( height + height / 2, width, CV_8UC1, map.data, width);
        cv::cvtColor(nv12, img, cv::COLOR_YUV2BGR_NV12);
        for (int i = 0; i < 720; i++) {
          if(h_result[i]!=-1) 
          {
            count++;
            //std::cout << "R" << i << " " << h_result[i] << ' ';
            cv::circle(img, cv::Point(h_result[i],i), 3, cv::Scalar( 255, 0, 0 ),cv::FILLED,cv::LINE_8);
          }
        }
        cv::imwrite("nv12.png", img);

    }
    gst_buffer_unmap(gstBuffer, &map);

    timer_5= cv::getTickCount();
    std::cout << "4-3-delta line analysis " << (timer_4 - timer_3) / cv::getTickFrequency() << std::endl;
    std::cout << "3-2-delta pre traitement " << (timer_3 - timer_2) / cv::getTickFrequency() << std::endl;
    std::cout << "2-1-delta mapping image " << (timer_2 - timer_1) / cv::getTickFrequency() << std::endl;
    std::cout << "4-2-delta " << (timer_4 - timer_2) / cv::getTickFrequency() << std::endl;
    std::cout << "4-1-delta " << (timer_4 - timer_1) / cv::getTickFrequency() << std::endl;
    std::cout << "4-0-delta " << (timer_4 - timer_0) / cv::getTickFrequency() << std::endl;
    std::cout << "0-6-delta " << (timer_0 - timer_6) / cv::getTickFrequency() << std::endl;
    timer_6= timer_0;

    return true;
  }

  int param_canny_min_;
  int param_canny_max_;
  int width;
  int height;
  int vh_;
  int u0_;
  double beta_x_;
  double beta_y_;

  int nb_loop;

  GstElement *pipeline = nullptr;
  GstElement *appsink = nullptr;

  VPIStream stream = nullptr;
  cudaStream_t cudaStream = nullptr;

  VPIPayload payload_eq;
  VPIPayload payload_canny;
  VPICannyEdgeDetectorParams canny_params;

  cv::Mat img;
  VPIImage vpiGray = nullptr, vpiCanny = nullptr;
  VPIImage vpiBlur = nullptr, vpiEq = nullptr;
  uint8_t *cudaGray = nullptr, *cudaCanny = nullptr;
  uint8_t *cudaBlur = nullptr, *cudaEq = nullptr;

  VPIImageData dataGray;
  VPIImageData dataBlur;
  VPIImageData dataEq;
  VPIImageData dataCanny;

  VPIImageWrapperParams wrapParams = {};
  VPIImageWrapperParams params;

  int *h_result = new int[height]; // Alloue sur le CPU
  int *h_init = new int[height](); // Rempli de zéros
  int *d_result;

  cudaPointerAttributes attr;

};

/***************************************/
/*    fin de la classe                 */
/***************************************/

GstFlowReturn CameraAnalysis::onNewSample( GstAppSink* appsink, gpointer user_data) {

    auto* camera = static_cast<CameraAnalysis*>(user_data);
    camera->timer_0 = cv::getTickCount();
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer)
    {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    GstCaps *caps = gst_sample_get_caps(sample);
    camera->createVPIImageDataFromGstBuffer(buffer,caps);
    return GST_FLOW_OK;
}

int main() {
  try {
    CameraAnalysis node;
    node.loop();
  } catch (const std::exception &e) {
    std::cerr << "Erreur fatale: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
