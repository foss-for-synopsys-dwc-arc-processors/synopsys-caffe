#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/ssd_decoder_layer.hpp"

namespace caffe {

template <typename Dtype>
void SSDDecoderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  objectness_score_ = detection_output_param.objectness_score();
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;
  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  eta_ = detection_output_param.nms_param().eta();
  CHECK_GT(eta_, 0.);
  CHECK_LE(eta_, 1.);
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
  const SaveOutputParameter& save_output_param =
      detection_output_param.save_output_param();
  output_directory_ = save_output_param.output_directory();
  if (!output_directory_.empty()) {
    if (boost::filesystem::is_directory(output_directory_)) {
      // boost::filesystem::remove_all(output_directory_);
    }
    if (!boost::filesystem::create_directories(output_directory_)) {
        LOG(WARNING) << "Failed to create directory: " << output_directory_;
    }
  }
  output_name_prefix_ = save_output_param.output_name_prefix();
  need_save_ = output_directory_ == "" ? false : true;
  output_format_ = save_output_param.output_format();
  if (save_output_param.has_label_map_file()) {
    string label_map_file = save_output_param.label_map_file();
    if (label_map_file.empty()) {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
      need_save_ = false;
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  } else {
    need_save_ = false;
  }
  if (save_output_param.has_name_size_file()) {
    string name_size_file = save_output_param.name_size_file();
    if (name_size_file.empty()) {
      // Ignore saving if there is no name_size_file provided.
      LOG(WARNING) << "Provide name_size_file if output results to files.";
      need_save_ = false;
    } else {
      std::ifstream infile(name_size_file.c_str());
      CHECK(infile.good())
          << "Failed to open name size file: " << name_size_file;
      // The file is in the following format:
      //    name height width
      //    ...
      string name;
      int height, width;
      while (infile >> name >> height >> width) {
        names_.push_back(name);
        sizes_.push_back(std::make_pair(height, width));
      }
      infile.close();
      if (save_output_param.has_num_test_image()) {
        num_test_image_ = save_output_param.num_test_image();
      } else {
        num_test_image_ = names_.size();
      }
      CHECK_LE(num_test_image_, names_.size());
    }
  } else {
    need_save_ = false;
  }
  has_resize_ = save_output_param.has_resize_param();
  if (has_resize_) {
    resize_param_ = save_output_param.resize_param();
  }
  name_count_ = 0;
  visualize_ = detection_output_param.visualize();
  if (visualize_) {
    visualize_threshold_ = 0.6;
    if (detection_output_param.has_visualize_threshold()) {
      visualize_threshold_ = detection_output_param.visualize_threshold();
    }
    data_transformer_.reset(
        new DataTransformer<Dtype>(this->layer_param_.transform_param(),
                                   this->phase_));
    data_transformer_->InitRand();
    save_file_ = detection_output_param.save_file();
  }
  bbox_preds_.ReshapeLike(*(bottom[0]));
  if (!share_location_) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }
  conf_permute_.ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void SSDDecoderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if (bbox_preds_.num() != bottom[0]->num() ||
      bbox_preds_.count(1) != bottom[0]->count(1)) {
    bbox_preds_.ReshapeLike(*(bottom[0]));
  }
  if (!share_location_ && (bbox_permute_.num() != bottom[0]->num() ||
      bbox_permute_.count(1) != bottom[0]->count(1))) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }

  num_priors_ = bottom[1]->height() / 4;
  CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_priors_);
  // Each row is a 4 dimension vector, which stores
  // [xmin, ymin, xmax, ymax]
  top_shape.push_back(4);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SSDDecoderLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* prior_data = bottom[1]->cpu_data();
  const int num = bottom[0]->num();

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  const bool clip_bbox = false;

DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
				share_location_, num_loc_classes_, background_label_id_,
				code_type_, variance_encoded_in_target_, clip_bbox,
				&all_decode_bboxes);

  Dtype* top_data = top[0]->mutable_cpu_data();

  int count = 0;
  //boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];

    int loc_label = -1; //share_location_ ? -1 : label;
    if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
      // Something bad happened if there are no predictions for current label.
      LOG(FATAL) << "Could not find location predictions for " << loc_label;
      continue;
    }
    const vector<NormalizedBBox>& bboxes =
        decode_bboxes.find(loc_label)->second;

    for (int j = 0; j < num_priors_; ++j) {
      const NormalizedBBox& bbox = bboxes[j];
      top_data[count * 4] = bbox.xmin();
      top_data[count * 4 + 1] = bbox.ymin();
      top_data[count * 4 + 2] = bbox.xmax();
      top_data[count * 4 + 3] = bbox.ymax();
      ++count;
    }
  }
}

INSTANTIATE_CLASS(SSDDecoderLayer);
REGISTER_LAYER_CLASS(SSDDecoder);

}  // namespace caffe
