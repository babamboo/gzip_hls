#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <chrono>
#include <fstream>
#include <string>

#include "CompareGzip.hpp"
#include "WriteGzip.hpp"
#include "crc32.hpp"
#include "gzipkernel.hpp"
#include "kernels.hpp"

#include "exception_handler.hpp"


using namespace sycl;

// The minimum file size of a file to be compressed.
// Any filesize less than this results in an error.
constexpr int minimum_filesize = kVec + 1;

bool help = false;

int CompressFile(queue &q, std::string &input_file, std::string &outfilename,
                 int iterations, bool report);

void Help(void) {
  // Command line arguments.
  // gzip [options] filetozip [options]
  // -h,--help                    : help

  // future options?
  // -p,performance : output perf metrics
  // -m,maxmapping=#  : maximum mapping size

  std::cout << "gzip filename [options]\n";
  std::cout << "  -h,--help                                : this help text\n";
  std::cout
      << "  -o=<filename>,--output-file=<filename>   : specify output file\n";
}

bool FindGetArg(std::string &arg, const char *str, int defaultval, int *val) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    int value = atoi(&arg.c_str()[strlen(str)]);
    *val = value;
    return true;
  }
  return false;
}

constexpr int kMaxStringLen = 40;

bool FindGetArgString(std::string &arg, const char *str, char *str_value,
                      size_t maxchars) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    const char *sptr = &arg.c_str()[strlen(str)];
    for (int i = 0; i < maxchars - 1; i++) {
      char ch = sptr[i];
      switch (ch) {
        case ' ':
        case '\t':
        case '\0':
          str_value[i] = 0;
          return true;
          break;
        default:
          str_value[i] = ch;
          break;
      }
    }
    return true;
  }
  return false;
}

size_t SyclGetExecTimeNs(event e) {
  size_t start_time =
      e.get_profiling_info<info::event_profiling::command_start>();
  size_t end_time =
      e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}

int main(int argc, char *argv[]) {
  std::string infilename = "";
  std::string outfilename;

  char str_buffer[kMaxStringLen] = {0};

  // Check the number of arguments specified
  if (argc != 3) {
    std::cerr << "Incorrect number of arguments. Correct usage: " << argv[0]
              << " <input-file> -o=<output-file>\n";
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      std::string sarg(argv[i]);
      if (std::string(argv[i]) == "-h") {
        help = true;
      }
      if (std::string(argv[i]) == "--help") {
        help = true;
      }

      FindGetArgString(sarg, "-o=", str_buffer, kMaxStringLen);
      FindGetArgString(sarg, "--output-file=", str_buffer, kMaxStringLen);
    } else {
      infilename = std::string(argv[i]);
    }
  }

  if (help) {
    Help();
    return 1;
  }

  try {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(selector, fpga_tools::exception_handler, prop_list);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<info::device::name>().c_str() 
              << std::endl;

    if (infilename == "") {
      std::cout << "Must specify a filename to compress\n\n";
      Help();
      return 1;
    }

    // next, check valid and acceptable parameter ranges.
    // if output filename not set, use the default
    // name, else use the name specified by the user
    outfilename = std::string(infilename) + ".gz";
    if (strlen(str_buffer)) {
      outfilename = std::string(str_buffer);
    }

    std::cout << "Launching High-Bandwidth DMA GZIP application";

#ifdef FPGA_EMULATOR
    CompressFile(q, infilename, outfilename, 1, true);
#elif FPGA_SIMULATOR
    CompressFile(q, infilename, outfilename, 2, true);
#else
    // warmup run - use this run to warmup accelerator. There are some steps in
    // the runtime that are only executed on the first kernel invocation but not
    // on subsequent invocations. So execute all that stuff here before we
    // measure performance (in the next call to CompressFile().
    CompressFile(q, infilename, outfilename, 1, false);
    // profile performance
    CompressFile(q, infilename, outfilename, 200, true);
#endif
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cerr << "If you are targeting the FPGA simulator, compile with "
                   "-DFPGA_SIMULATOR.\n";
    }
    std::terminate();
  }
  return 0;
}

struct KernelInfo {
  struct GzipOutInfo *gzip_out_buf;
  unsigned *current_crc;
  char *pobuf;
  char *pibuf;

  uint32_t refcrc;

  size_t file_size;
  int iteration;
  bool last_block;
};

// returns 0 on success, otherwise a non-zero failure code.
int CompressFile(queue &q, std::string &input_file, std::string &outfilename,
                 int iterations, bool report) {
  size_t isz;
  char *pinbuf;

  // Read the input file
  std::string device_string =
      q.get_device().get_info<info::device::name>().c_str();

  // If the device is supports USM allocations, we pre-pin some buffers to
  // improve DMA performance, which is needed to
  // achieve peak kernel throughput. Pre-pinning is
  // only supported on the USM capable BSPs. It's not
  // needed on non-USM capabale BSPs to achieve peak performance.
#ifdef FPGA_SIMULATOR
  bool prepin = false;
#else
  bool prepin = q.get_device().has(aspect::usm_host_allocations);
#endif

  // padding for the input and output buffers to deal with granularity of
  // kernel reads and writes
  constexpr size_t kInOutPadding = 16 * kVec;
  
  std::ifstream file(input_file,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    isz = file.tellg();
    if (prepin) {
      pinbuf = (char *)malloc_host(
          isz + kInOutPadding, q.get_context());  // Pre-pin the buffer, for faster DMA
    } else {                      // throughput, using malloc_host().
      pinbuf = new char[isz + kInOutPadding];
    }
    file.seekg(0, std::ios::beg);
    file.read(pinbuf, isz);
    file.close();
  } else {
    std::cout << "Error: cannot read specified input file\n";
    return 1;
  }

  if (isz < minimum_filesize) {
    std::cout << "Minimum filesize for compression is " << minimum_filesize
              << "\n";
    return 1;
  }

  int buffers_count = iterations;

  // Create an array of kernel info structures and create buffers for kernel
  // input/output. The buffers are re-used between iterations, but enough 
  // disjoint buffers are created to support double-buffering.
  struct KernelInfo *kinfo;
  kinfo = (struct KernelInfo *)malloc(sizeof(struct KernelInfo) * buffers_count);
  if (kinfo == NULL) {
    std::cout << "Cannot allocate kernel info buffer.\n";
    return 1;
  }
  for (int i = 0; i < buffers_count; i++) {
    kinfo[i].file_size = isz;
    // Allocating slightly larger buffers (+ 16 * kVec) to account for
    // granularity of kernel writes
    int outputSize =
      ((isz + kInOutPadding) < kMinBufferSize) ? kMinBufferSize
      : (isz + kInOutPadding);
    kinfo[i].last_block = true;
    kinfo[i].iteration = i;

    kinfo[i].gzip_out_buf =
      i >= 3 ? kinfo[i - 3].gzip_out_buf
      : new struct GzipOutInfo[kMinBufferSize];
    kinfo[i].current_crc = i >= 3
      ? kinfo[i - 3].current_crc
      : new unsigned[kMinBufferSize];
    kinfo[i].pibuf = i >= 3
      ? kinfo[i - 3].pibuf
      : pinbuf;
    kinfo[i].pobuf =
      i >= 3 ? kinfo[i - 3].pobuf
      : new char[outputSize];
  }


  // Create events for the various parts of the execution so that we can profile
  // their performance.
  std::vector<event> e_input_dma ; // Input to the GZIP engine. This is a transfer from host to device.
  std::vector<event> e_output_dma; // Output from the GZIP engine. This is transfer from device to host.
  std::vector<event> e_crc_dma   ; // Transfer CRC from device to host
  std::vector<event> e_size_dma  ; // Transfer compressed file size from device to host
  std::vector<event> e_k_crc     ; // CRC kernel
  std::vector<event> e_k_lz      ; // LZ77 kernel
  std::vector<event> e_k_huff    ; // Huffman Encoding kernel

  e_input_dma.resize(buffers_count);
  e_output_dma.resize(buffers_count);
  e_crc_dma.resize(buffers_count);
  e_size_dma.resize(buffers_count);
  e_k_crc.resize(buffers_count);
  e_k_lz.resize(buffers_count);
  e_k_huff.resize(buffers_count);


  /*************************************************/
  /* Main loop where the actual execution happens  */
  /*************************************************/
  for (int i = 0; i < buffers_count; i++) {

    /************************************/
    /************************************/
    /*         LAUNCH GZIP ENGINE       */
    /************************************/
    /************************************/
    SubmitGzipTasks(q, kinfo[i].file_size, kinfo[i].pibuf,
		    kinfo[i].pobuf, kinfo[i].gzip_out_buf,
		    kinfo[i].current_crc, kinfo[i].last_block,
		    e_k_crc, e_k_lz, e_k_huff,
		    i);
  }


  // Check the compressed file size from each iteration. Make sure the size is actually
  // less-than-or-equal to the input size. Also calculate the remaining CRC.
  size_t compressed_sz;

  compressed_sz = 0;
  for (int i = 0; i < buffers_count; i++) {
    if (kinfo[i].gzip_out_buf[0].compression_sz > kinfo[i].file_size) {
      std::cerr << "Unsupported: compressed file larger than input file( "
		<< kinfo[i].gzip_out_buf[0].compression_sz << " )\n";
      return 1;
    }
    // The majority of the CRC is calculated by the CRC kernel on the FPGA. But the kernel
    // operates on quantized chunks of input data, so any remaining input data, that falls
    // outside the quanta, is included in the overall CRC calculation via the following 
    // function that runs on the host. The last argument is the running CRC that was computed
    // on the FPGA.
    kinfo[i].current_crc[0] =
      Crc32(kinfo[i].pibuf, kinfo[i].file_size,
	    kinfo[i].current_crc[0]);
    // Accumulate the compressed size across all iterations. Used to 
    // compute compression ratio later.
    compressed_sz += kinfo[i].gzip_out_buf[0].compression_sz;
  }


  // delete the file mapping now that all kernels are complete, and we've
  // snapped the time delta
  if (prepin) {
    free(pinbuf, q.get_context());
  } else {
    delete pinbuf;
  }

  // Write the output compressed data from the first iteration of each engine, to a file.
  if (report && WriteBlockGzip(input_file, outfilename, kinfo[0].pobuf,
			       kinfo[0].gzip_out_buf[0].compression_sz,
			       kinfo[0].file_size, kinfo[0].current_crc[0])) {
    std::cout << "FAILED\n";
    return 1;
  }        

  // Decompress the output from engine-0 and compare against the input file. Only engine-0's
  // output is verified since all engines are fed the same input data.
  if (report && CompareGzipFiles(input_file, outfilename)) {
    std::cout << "FAILED\n";
    return 1;
  }

  /*
  // Generate throughput report
  // First gather all the execution times.
  size_t time_k_crc;
  size_t time_k_lz;
  size_t time_k_huff;
  size_t time_input_dma;
  size_t time_output_dma;

    time_k_crc = 0;
    time_k_lz = 0;
    time_k_huff = 0;
    time_input_dma = 0;
    time_output_dma = 0;
    for (int i = 0; i < buffers_count; i++) {
      time_k_crc       += SyclGetExecTimeNs(e_k_crc[i]);
      time_k_lz        += SyclGetExecTimeNs(e_k_lz[i]);
      time_k_huff      += SyclGetExecTimeNs(e_k_huff[i]);
      time_input_dma   += SyclGetExecTimeNs(e_input_dma[i]);
      time_output_dma  += SyclGetExecTimeNs(e_output_dma[i]);
    }

  */
  if (report) {
    double compression_ratio =
        (double)((double)compressed_sz / (double)isz / iterations);
    /*
#ifdef FPGA_EMULATOR
#elif FPGA_SIMULATOR
#else
    std::cout << "Throughput: " << gbps << " GB/s\n\n";
      std::cout << "TP breakdown for engine #" << eng << " (GB/s)\n";
      std::cout << "CRC = " << iterations * isz / (double)time_k_crc
                << "\n";
      std::cout << "LZ77 = " << iterations * isz / (double)time_k_lz
                << "\n";
      std::cout << "Huffman Encoding = "
                << iterations * isz / (double)time_k_huff << "\n";
      std::cout << "DMA host-to-device = "
                << iterations * isz / (double)time_input_dma << "\n";
      std::cout << "DMA device-to-host = "
                << iterations * isz / (double)time_output_dma << "\n\n";
#endif
    */
    std::cout << "Compression Ratio " << compression_ratio * 100 << "%\n";
  }

  // Cleanup anything that was allocated by this routine.
  free(kinfo);

  if (report) std::cout << "PASSED\n";
  return 0;
}
