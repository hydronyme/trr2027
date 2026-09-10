// camera_analysis.cpp
//
// Conversion C++ de la classe Python CameraAnalysis.
// Pipeline CPU : OpenCV (cv::cvtColor / GaussianBlur / equalizeHist / Canny)
// Pipeline GPU : NVIDIA VPI (backend CUDA), lecture CPU via vpiImageLockData.
//
// NOTE IMPORTANTE :
// L'API VPI (noms de fonctions, signatures exactes, en-têtes) a évolué entre
// les versions 1.x / 2.x / 3.x, elles-mêmes liées à la version de JetPack.
// Le code ci-dessous est écrit pour l'API "C" classique de VPI (vpiSubmitXxx /
// VPIImage / VPIStream), qui est celle réellement exposée en C++ (le module
// Python `vpi` est un wrapper au-dessus de cette API C). Vérifiez les noms
// exacts des en-têtes/fonctions dans /opt/nvidia/vpi*/include pour votre
// JetPack 7.2 et ajustez si besoin (ils changent peu d'une version à l'autre,
// mais mieux vaut confirmer).



#include <opencv2/opencv.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudarational.hpp>

#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for applyColorMap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <cuda_runtime.h>
#include <nvbufsurface.h>


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/CUDAInterop.h>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Context.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/BoxFilter.h>
#include <vpi/algo/EqualizeHist.h>
#include <vpi/algo/CannyEdges.h>



#include <cstdio>
#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

//#include <gst/gst.h>
//#include <gst/app/gstappsink.h>

#include "line_analysis.cuh"

#define MAX_BORDERS 512

// ---------------------------------------------------------------------
// Petit helper pour transformer un code d'erreur VPI en exception C++.
// ---------------------------------------------------------------------
#define CHECK_VPI(STMT)                                                      \
    do {                                                                     \
        VPIStatus status = (STMT);                                           \
        if (status != VPI_SUCCESS) {                                         \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];                      \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));                 \
            std::ostringstream oss;                                          \
            oss << "Erreur VPI (" << vpiStatusGetName(status) << "): "       \
                << buffer;                                                   \
            throw std::runtime_error(oss.str());                             \
        }                                                                    \
    } while (0)

 #define CHECK_CUDA_STATUS(STMT)                 \
     do                                          \
     {                                           \
         cudaError_t status = (STMT);            \
         if (status != cudaSuccess)              \
         {                                       \
             std::ostringstream ss;              \
             ss << cudaGetErrorString(status);   \
             throw std::runtime_error(ss.str()); \
         }                                       \
     } while (0);

class CameraAnalysis {
public:
    CameraAnalysis()
        : param_canny_min_(150),
          param_canny_max_(200),
          width_(1280),
          height_(720),
          vh_(200),
          u0_(640),
          beta_x_(0.13646387832699614),
          beta_y_(67.19999999999997) {
        initVPI();
        camera_ = initCamera();
        defineLines();
    }

    ~CameraAnalysis() {
        if (stream_) vpiStreamDestroy(stream_);
        //if (payload_eq) vpiStreamDestroy(payload_eq);
        //if (payload_canny) vpiStreamDestroy(payload_canny);
    }

    void loop();

private:
    struct TargetResult {
        int u = -1;
        int v = 0;
        double x = 0.0;
        double y = 0.0;
        bool found = false;
    };

    // ------------------------ Initialisation ------------------------

    cv::VideoCapture initCamera() {
        std::string gst =
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=(int)" + std::to_string(width_) +
            ", height=(int)" + std::to_string(height_) +
            ", framerate=(fraction)60/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)" + std::to_string(width_) +
            ", height=(int)" + std::to_string(height_) +
            ", format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink";
        return cv::VideoCapture(gst, cv::CAP_GSTREAMER);
    }

    void initVPI() {
	CHECK_CUDA_STATUS(cudaStreamCreate(&cudaStream));
        CHECK_VPI(vpiStreamCreateWrapperCUDA(cudaStream, 0, &stream_));
        vpiCreateEqualizeHist(VPI_BACKEND_CUDA, VPI_IMAGE_FORMAT_U8, &payload_eq);
        vpiCreateCannyEdgeDetector(VPI_BACKEND_CUDA, width_, height_, &payload_canny);
        vpiInitCannyEdgeDetectorParams(&params);
    }


    VPIImageData GetGenericPLData(int width_, int height_, VPIImageBufferType bufferType, VPIByte *pBase)
	{
	    VPIImageBufferPitchLinear imgData = {};
	    imgData.format                    = VPI_IMAGE_FORMAT_U8;
	    imgData.numPlanes                 = 1;
	    imgData.planes[0].width           = width_;
	    imgData.planes[0].height          = height_;
	    imgData.planes[0].pitchBytes      = sizeof(uint8_t) * width_;
	    imgData.planes[0].pixelType       = VPI_PIXEL_TYPE_INVALID;
	    imgData.planes[0].offsetBytes     = 0;
	    imgData.planes[0].pBase           = pBase;
	    VPIImageData vpiImgData           = {};
	    vpiImgData.bufferType             = bufferType;
	    vpiImgData.buffer.pitch           = imgData;
	    return vpiImgData;
	}


	void defineLines() {
		int k = 0;
		int v0 = (vh_ + height_) / 2;
		while (true) {
		    int v = v0 + k * 10;
		    if (v > height_) break;
		    target_line_.push_back(v);
		    ++k;
		    v = v0 - k * 10;
		    if (v < vh_) break;
		    target_line_.push_back(v);
		}
		for (int i = 0; i < 10; ++i) {
		    int v = height_ - k * 10 - 1;
		    error_line_.push_back(v);
		}
	    }



    // --------------------- Analyse d'une ligne ----------------------
    //
    // Cherche une bande blanche (>200) entourée de deux bandes noires (<80)
    // dans le profil `canny`/`gray` à la ligne v. Retourne l'indice colonne
    // du centre de la bande blanche, ou -1 si rien trouvé.
    int analyseLine(const cv::Mat& gray, const cv::Mat& canny, int v) {
        const uint8_t* edges = canny.ptr<uint8_t>(v);
        const uint8_t* color = gray.ptr<uint8_t>(v);
        const int w = canny.cols;

        // 1. indices des pixels de contour (equiv. np.flatnonzero(edges == 255))
        std::vector<int> bord_all;
        bord_all.reserve(64);
        for (int i = 0; i < w; ++i) {
            if (edges[i] == 255) bord_all.push_back(i);
        }
        if (static_cast<int>(bord_all.size()) < 3) return -1;

        // 2. filtre les contours trop rapprochés
        //    (equiv. bord[np.ediff1d(bord, to_end=20) > 9])
        std::vector<int> bord;
        bord.reserve(bord_all.size());
        for (size_t i = 0; i < bord_all.size(); ++i) {
            int diff = (i + 1 < bord_all.size())
                           ? (bord_all[i + 1] - bord_all[i])
                           : 20;
            if (diff > 9) bord.push_back(bord_all[i]);
        }
        const int n = static_cast<int>(bord.size());
        if (n < 3) return -1;

        // 3. delta entre bords consécutifs
        std::vector<int> delta(n - 1);
        for (int i = 0; i < n - 1; ++i) delta[i] = bord[i + 1] - bord[i];

        // 4. couleur moyenne par segment, via somme cumulée (évite de
        //    recalculer une moyenne "from scratch" pour chaque segment)
        std::vector<int64_t> csum(w + 1, 0);
        for (int i = 0; i < w; ++i) csum[i + 1] = csum[i] + color[i];

        std::vector<int> mean_color(n - 1);
        for (int i = 0; i < n - 1; ++i) {
            mean_color[i] =
                static_cast<int>((csum[bord[i + 1]] - csum[bord[i]]) / delta[i]);
        }
        if (static_cast<int>(mean_color.size()) < 3) return -1;

        // 5. recherche du premier triplet noir/blanc/noir avec deltas cohérents
        for (int idx = 1; idx < static_cast<int>(mean_color.size()) - 1; ++idx) {
            if (mean_color[idx] > 200 && mean_color[idx - 1] < 80 &&
                mean_color[idx + 1] < 80 && std::abs(delta[idx] - delta[idx - 1]) < delta[idx] / 2 &&
                std::abs(delta[idx] - delta[idx + 1]) < delta[idx] / 2) {
                return (bord[idx] + bord[idx + 1]) / 2;
            }
        }
        return -1;
    }

    TargetResult findTargetBand(const cv::Mat& gray, const cv::Mat& canny) {
        for (int v : target_line_) {
            int u = analyseLine(gray, canny, v);
            if (u != -1) {
                double x = beta_x_ * (u - u0_) / static_cast<double>(v - vh_);
                double y = beta_y_ / static_cast<double>(v - vh_);
                return {u, v, x, y, true};
            }
        }
        return {-1, 0, 0.0, 0.0, false};
    }

    double findErrorBand(const cv::Mat& gray, const cv::Mat& canny) {
        for (int v : error_line_) {
            int u = analyseLine(gray, canny, v);
            if (u != -1) {
                return u / static_cast<double>(width_) - 0.5;
            }
        }
        return -9999.0;
    }

    // ----------------------------- Etat ------------------------------

    int param_canny_min_;
    int param_canny_max_;
    int width_;
    int height_;
    int vh_;
    int u0_;
    double beta_x_;
    double beta_y_;

    cv::VideoCapture camera_;
    std::vector<int> target_line_;
    std::vector<int> error_line_;

    VPIStream stream_ = nullptr;
    cudaStream_t cudaStream = nullptr;

    VPIPayload payload_eq;
    VPIPayload payload_canny;
    VPICannyEdgeDetectorParams params;

};

void CameraAnalysis::loop() {
	// Allocate pointers to be wrapped and create wrapped images
        cv::Mat img;
	VPIImage vpiGray = nullptr, vpiCanny = nullptr;
	VPIImage vpiBlur = nullptr, vpiEq = nullptr;
	uint8_t *cudaGray = nullptr, *cudaCanny = nullptr;
	uint8_t *cudaBlur = nullptr, *cudaEq = nullptr;
	VPIImageWrapperParams wrapParams = {};
	wrapParams.colorSpec             = VPI_COLOR_SPEC_DEFAULT;

	int *h_result = new int[height_]; // Alloue sur le CPU
	int *d_result;
	cudaMalloc(&d_result, height_ * sizeof(int)); // Alloue sur le GPU

	    cudaPointerAttributes attr;

	// Optionnel : initialise d_result avec des zéros (si nécessaire)
	int *h_init = new int[height_](); // Rempli de zéros
	cudaMemcpy(d_result, h_init, height_ * sizeof(int), cudaMemcpyHostToDevice);
	//delete[] h_init; // Libère l'initialisation temporaire

        // Allocate CUDA pitch
        CHECK_CUDA_STATUS(cudaMalloc(&cudaGray, width_ * height_ * sizeof(uint8_t)));
        CHECK_CUDA_STATUS(cudaMalloc(&cudaBlur, width_ * height_ * sizeof(uint8_t)));
        CHECK_CUDA_STATUS(cudaMalloc(&cudaEq, width_ * height_ * sizeof(uint8_t)));
        CHECK_CUDA_STATUS(cudaMalloc(&cudaCanny, width_ * height_ * sizeof(uint8_t)));

	// Alloue avec pitch
	/*
	size_t pitch;
	CHECK_CUDA_STATUS(cudaMallocPitch(&cudaGray, &pitch, width_ * sizeof(uint8_t), height_));
	CHECK_CUDA_STATUS(cudaMallocPitch(&cudaBlur, &pitch, width_ * sizeof(uint8_t), height_));
	CHECK_CUDA_STATUS(cudaMallocPitch(&cudaEq, &pitch, width_ * sizeof(uint8_t), height_));
	CHECK_CUDA_STATUS(cudaMallocPitch(&cudaCanny, &pitch, width_ * sizeof(uint8_t), height_));
	*/

        // Wrap CUDA pitches directly in VPIImage
        VPIImageData dataGray  = GetGenericPLData(width_, height_, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaGray);
        VPIImageData dataBlur  = GetGenericPLData(width_, height_, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaBlur);
        VPIImageData dataEq    = GetGenericPLData(width_, height_, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaEq);
        VPIImageData dataCanny = GetGenericPLData(width_, height_, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, cudaCanny);

	VPIImage vpiImgIn = nullptr;

	// 1. Allocation d'une image VPI forcée en mémoire Device CUDA
	//CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_BGR8, VPI_BACKEND_CUDA, &vpiImgIn));
	//CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &bufGray));

        CHECK_VPI(vpiImageCreateWrapper(&dataGray, &wrapParams, VPI_BACKEND_CUDA, &vpiGray));
        CHECK_VPI(vpiImageCreateWrapper(&dataBlur, &wrapParams, VPI_BACKEND_CUDA, &vpiBlur));
        CHECK_VPI(vpiImageCreateWrapper(&dataEq, &wrapParams, VPI_BACKEND_CUDA, &vpiEq));
        CHECK_VPI(vpiImageCreateWrapper(&dataCanny, &wrapParams, VPI_BACKEND_CUDA, &vpiCanny));

        //const std::string filename = "./road/cam_2974997494746.jpg";
        const std::string filename = "./road/1000032674.jpg";
        cv::Mat gray, blur, eq, canny;

	TargetResult cpu_result;

	/*
	 gst_init(&argc, &argv);

    // 1. Pipeline GStreamer avec décodage matériel NVDEC
    // filesrc -> jpegparse -> nvv4l2decoder (NVDEC) -> nvvideoconvert (RAM GPU NVMM) -> appsink
    std::string pipeline =
        "filesrc location=image.jpg ! jpegparse ! nvv4l2decoder ! "
        "nvvideoconvert ! video/x-raw(memory:NVMM), format=NV12 ! appsink name=sink sync=false";

    GError* error = nullptr;
    GstElement* gstPipeline = gst_parse_launch(pipeline.c_str(), &error);
    if (!gstPipeline) {
        std::cerr << "Erreur pipeline GStreamer: " << error->message << std::endl;
        return -1;
    }

    GstElement* appsink = gst_bin_get_by_name(GST_BIN(gstPipeline), "sink");
    gst_element_set_state(gstPipeline, GST_STATE_PLAYING);

    // 2. Capture du buffer GPU NVMM décodé
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    GstBuffer* buffer = gst_sample_get_buffer(sample);

    // 3. Encapsulation directe du buffer NVMM GPU dans VPIImage (Zero-Copy, pure mémoire Device)
    VPIImage vpiImgIn = nullptr;
    vpiImageCreateGStreamerBufferWrapper(buffer, 0, &vpiImgIn);

    // 4. Allocation du buffer de sortie VPI en mémoire GPU CUDA dédiée
    VPIImage vpiImgOut = nullptr;
    VPIImageData imgDataIn;
    vpiImageLockData(vpiImgIn, VPI_LOCK_READ, VPI_IMAGE_BUFFER_FORMAT_UNKNOWN, &imgDataIn);

    vpiImageCreate(imgDataIn.size.width, imgDataIn.size.height, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &vpiImgOut);
    vpiImageUnlockData(vpiImgIn);
    */
    while (true) {
        int64 s0 = cv::getTickCount();
        //camera_.read(img);
	img = cv::imread(filename);
	cv::resize( img, img, cv::Size(width_, height_), 0.0, 0.0, cv::INTER_LINEAR);

        int64 e0 = cv::getTickCount();
        double d0 = (e0 - s0) / cv::getTickFrequency();
        std::cout << "0- " << d0 << std::endl;

        // ------------------------- Pipeline CPU -------------------------
        int64 s1 = cv::getTickCount();

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        cv::equalizeHist(blur, eq);
        cv::Canny(eq, canny, param_canny_min_, param_canny_max_);
        cpu_result = findTargetBand(gray, canny);
	std::cout << cpu_result.u << "\n";
	std::cout << cpu_result.v << "\n";
	std::cout << cpu_result.x << "\n";
	std::cout << cpu_result.y << "\n";
	std::cout << cpu_result.found << "\n";

        int64 e1 = cv::getTickCount();
        double d1 = (e1 - s1) / cv::getTickFrequency();
        std::cout << "1- " << d1 << std::endl;

        // ------------------------- Pipeline GPU -------------------------

        int64 s2 = cv::getTickCount();

	// 1. Allocation de vpiImgIn STRICTEMENT en mémoire CUDA Device (cudaMemoryTypeDevice)
    VPIImage vpiIn = nullptr;
    vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_BGR8, VPI_BACKEND_CUDA, &vpiIn);

    // 2. Verrouillage de l'image VPI en demandant spécifiquement le layout CUDA Pitch Linear
    VPIImageData dataIn;
    vpiImageLockData(vpiIn, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dataIn);

    // 3. Extraction de l'adresse GPU Device
    // Pour une image CUDA_PITCH_LINEAR à un seul plan (BGR8) :
    void* d_ptr = dataIn.buffer.pitch.planes[0].pBase;
    size_t dst_pitch = dataIn.buffer.pitch.planes[0].pitchBytes;


    // VÉRIFICATION DE SÉCURITÉ : Contrôle que la mémoire est bien cudaMemoryTypeDevice
    cudaPointerGetAttributes(&attr, d_ptr);
    if (attr.type == cudaMemoryTypeDevice) {
        std::cout << "--> Allocation confirmée : cudaMemoryTypeDevice (Mémoire GPU pure)" << std::endl;
    }

    // 4. Copie 2D de la RAM Host (cv::Mat CPU) vers la RAM GPU (d_ptr)
    // sans passer par une matrice cv::Mat temporaire
    size_t width_bytes = img.cols * img.elemSize(); // Largeur en octets (cols * 3)
    size_t src_pitch = img.step;                    // Pitch de l'image CPU

    cudaMemcpy2D(
        d_ptr,              // Destination : Pointeur GPU VPI
        dst_pitch,          // Alignement (Pitch) GPU
        img.data,           // Source : Pointeur CPU OpenCV
        src_pitch,          // Pitch CPU
        width_bytes,        // Largeur d'une ligne
        img.rows,           // Nombre de lignes
        cudaMemcpyHostToDevice
    );

    // 5. Déverrouillage : vpiImgIn est maintenant chargé et prêt pour des calculs ultra-rapides
    vpiImageUnlock(vpiIn);


        int64 e2 = cv::getTickCount();
        double d2 = (e2 - s2) / cv::getTickFrequency();
        std::cout << "2- " << d2 << std::endl;

        int64 s3 = cv::getTickCount();
        // 4. Conversion BGR8 (CPU/OpenCV) vers Niveau de gris (GPU)
	// Exécution asynchrone sur l'accélérateur CUDA de la Jetson
	CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, vpiIn, vpiGray, NULL));
        CHECK_VPI(vpiSubmitBoxFilter(stream_, VPI_BACKEND_CUDA, vpiGray, vpiBlur, 5, 5, VPI_BORDER_ZERO));
        CHECK_VPI(vpiSubmitEqualizeHist(stream_, VPI_BACKEND_CUDA, payload_eq, vpiBlur, vpiEq));
	CHECK_VPI(vpiSubmitCannyEdgeDetector(stream_, VPI_BACKEND_CUDA, payload_canny, vpiEq, vpiCanny, param_canny_max_, param_canny_min_, 255, 0, &params));

        // Un seul point de synchronisation pour toute la chaîne GPU.
        CHECK_VPI(vpiStreamSync(stream_));

        int64 e3 = cv::getTickCount();
        double d3 = (e3 - s3) / cv::getTickFrequency();
        std::cout << "3- " << d3 << std::endl;
        std::cout << "t- " << (d0 + d1) << " " << (1.0 / 60.0) << std::endl;
	break;

	cudaPointerGetAttributes(&attr, vpiImgIn);
	printf("Pointeur sur le GPU ? %d\n", attr.type == cudaMemoryTypeDevice);
	cudaPointerGetAttributes(&attr, cudaGray);
	printf("Pointeur sur le GPU ? %d\n", attr.type == cudaMemoryTypeDevice);

	/*
	vpiImageLockData(bufGray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dataGray);
	uint8_t *cudaGray = reinterpret_cast<uint8_t *>(dataGray.buffer.pitch.planes[0].pBase);
	int pitch = dataGray.buffer.pitch.planes[0].pitchBytes;
	CHECK_VPI(vpiImageUnlock(bufGray));
	*/
        std::cout << "conversion" << std::endl;

	submitLineAnalysis(cudaGray, cudaCanny, width_, height_, cudaStream, d_result);

        vpiImageUnlock(vpiGray);
        vpiImageUnlock(vpiCanny);

        CHECK_VPI(vpiStreamSync(stream_));
	cudaMemcpy(h_result, d_result, height_ * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 720; i++) {
          std::cout << "R" << i << " " << h_result[i] << ' ';
        }
        std::cout << std::endl;

	cv::Mat canny(height_, width_, CV_8UC1);
	CHECK_CUDA_STATUS(cudaMemcpy(canny.data, cudaCanny, width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	imwrite("canny.png", canny);

	cv::Mat gray2(height_, width_, CV_8UC1);
	CHECK_CUDA_STATUS(cudaMemcpy(gray2.data, cudaGray, width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	imwrite("gray.png", gray2);
	break;

    }
}

int main() {
    try {
        CameraAnalysis node;
        node.loop();
    } catch (const std::exception& e) {
        std::cerr << "Erreur fatale: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
