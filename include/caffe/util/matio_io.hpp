#if !defined(_MSC_VER)
#ifndef CAFFE_UTIL_MATIO_IO_H_
#define CAFFE_UTIL_MATIO_IO_H_

#if defined(_MSC_VER)
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif
#include <string>

#include "google/protobuf/message.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using ::google::protobuf::Message;

template <typename Dtype>
void ReadBlobFromMat(const char *fname, Blob<Dtype>* blob);

template <typename Dtype>
void WriteBlobToMat(const char *fname, bool write_diff,
   Blob<Dtype>* blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
#endif
