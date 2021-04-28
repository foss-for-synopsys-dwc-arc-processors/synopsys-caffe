#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  ratio_permute_ = detection_output_param.ratio_permute();
  no_permute_ = detection_output_param.no_permute();
  nbottom_ = detection_output_param.nbottom();
  ratio0_ = detection_output_param.ratio0();
  ratio1_ = detection_output_param.ratio1();
  ratio2_ = detection_output_param.ratio2();
  ratio3_ = detection_output_param.ratio3();
  ratio4_ = detection_output_param.ratio4();
  ratio5_ = detection_output_param.ratio5();
  // TFLite_Detection_PostProcess related params
  tflite_detection_ = detection_output_param.tflite_detection();
  tflite_use_regular_nms_ = detection_output_param.tflite_use_regular_nms();
  max_classes_per_detection_ =
      detection_output_param.max_classes_per_detection();
  scale_xywh_.clear();
  std::copy(detection_output_param.scale_xywh().begin(),
            detection_output_param.scale_xywh().end(),
            std::back_inserter(scale_xywh_));
  CHECK_EQ(tflite_use_regular_nms_, 0)
      << "tflite_use_regular_nms is not supported yet!";

  objectness_score_ = detection_output_param.objectness_score();
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold()
                              ? detection_output_param.confidence_threshold()
                              : -FLT_MAX;
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
  const SaveOutputParameter &save_output_param =
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
      CHECK(infile.good()) << "Failed to open name size file: "
                           << name_size_file;
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
    data_transformer_.reset(new DataTransformer<Dtype>(
        this->layer_param_.transform_param(), this->phase_));
    data_transformer_->InitRand();
    save_file_ = detection_output_param.save_file();
  }
  if (bottom.size() < nbottom_) {
    conf_concat_ = true;
    loc_concat_ = true;
    priorbox_concat_ = true;
  } else if (bottom.size() >= nbottom_ && bottom.size() < 2 * nbottom_ + 1) {
    conf_concat_ = false;
    loc_concat_ = true;
    priorbox_concat_ = true;
  } else if (bottom.size() >= 2 * nbottom_ + 1 &&
             bottom.size() < 3 * nbottom_) {
    conf_concat_ = false;
    loc_concat_ = false;
    priorbox_concat_ = true;
  } else // ==3*nbottom_
  {
    conf_concat_ = false;
    loc_concat_ = false;
    priorbox_concat_ = false;
  }

  if (conf_concat_ && loc_concat_) {
    bbox_preds_.ReshapeLike(*(bottom[0]));
    if (!share_location_) {
      bbox_permute_.ReshapeLike(*(bottom[0]));
    }
    conf_permute_.ReshapeLike(*(bottom[1]));
  }
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  if (need_save_) {
    CHECK_LE(name_count_, names_.size());
    if (name_count_ % num_test_image_ == 0) {
      // Clean all outputs.
      if (output_format_ == "VOC") {
        boost::filesystem::path output_directory(output_directory_);
        for (map<int, string>::iterator it = label_to_name_.begin();
             it != label_to_name_.end(); ++it) {
          if (it->first == background_label_id_) {
            continue;
          }
          std::ofstream outfile;
          boost::filesystem::path file(output_name_prefix_ + it->second +
                                       ".txt");
          boost::filesystem::path out_file = output_directory / file;
          outfile.open(out_file.string().c_str(), std::ofstream::out);
        }
      }
    }
  }

  if (conf_concat_ && loc_concat_) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    if (bbox_preds_.num() != bottom[0]->num() ||
        bbox_preds_.count(1) != bottom[0]->count(1)) {
      bbox_preds_.ReshapeLike(*(bottom[0]));
    }
    if (!share_location_ && (bbox_permute_.num() != bottom[0]->num() ||
                             bbox_permute_.count(1) != bottom[0]->count(1))) {
      bbox_permute_.ReshapeLike(*(bottom[0]));
    }
    if (conf_permute_.num() != bottom[1]->num() ||
        conf_permute_.count(1) != bottom[1]->count(1)) {
      conf_permute_.ReshapeLike(*(bottom[1]));
    }

    if (priorbox_concat_)
      num_priors_ = bottom[2]->height() / 4;
    CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->channels())
        << "Number of priors must match number of location predictions.";
    CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
        << "Number of priors must match number of confidence predictions.";
  }
  // TODO: currently only consider the case when both are separate into nbottom_
  // layers
  else if (!conf_concat_ && !loc_concat_) {
    for (int n = 0; n < nbottom_; n++) {
      CHECK_EQ(bottom[n]->num(), bottom[n + nbottom_]->num());
      if (no_permute_) {
        CHECK_EQ(bottom[n]->channels(), num_classes_);
        CHECK_EQ(bottom[n]->height(),
                 bottom[n + nbottom_]->channels() / 4 / num_loc_classes_);
        CHECK_EQ(bottom[n]->width(), bottom[n + nbottom_]->height() *
                                         bottom[n + nbottom_]->width());
        collect_ratios_.push_back(bottom[n + nbottom_]->channels() / 4 /
                                  num_loc_classes_);
      }
    }
    int sum_conf = 0;
    int sum_loc = 0;
    for (int n = 0; n < nbottom_; n++) {
      sum_conf += bottom[n]->channels();
      sum_loc += bottom[n + nbottom_]->channels();
    }
    if (priorbox_concat_)
      num_priors_ = bottom[2 * nbottom_]->height() / 4;
    else {
      num_priors_ = 0;
      for (int n = 0; n < nbottom_; n++)
        num_priors_ += bottom[n + 2 * nbottom_]->height() / 4;
    }
    if (!no_permute_) {
      CHECK_EQ(num_priors_ * num_loc_classes_ * 4, sum_loc)
          << "Number of priors must match number of location predictions.";
      CHECK_EQ(num_priors_ * num_classes_, sum_conf)
          << "Number of priors must match number of confidence predictions.";
    }
  }
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
  if (tflite_detection_) {
    vector<int> top_shape;
    top_shape.push_back(max_classes_per_detection_ * keep_top_k_);
    top_shape.push_back(7);
    top[0]->Reshape(top_shape);
  }
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int num = bottom[0]->num();
  const Dtype *arm_conf_data = NULL;
  const Dtype *arm_loc_data = NULL;
  vector<LabelBBox> all_arm_loc_preds;
  if (bottom.size() >= 4 && conf_concat_) {
    arm_conf_data = bottom[3]->cpu_data();
  }
  if (bottom.size() >= 5 && loc_concat_) {
    arm_loc_data = bottom[4]->cpu_data();
    GetLocPredictions(arm_loc_data, num, num_priors_, num_loc_classes_,
                      share_location_, &all_arm_loc_preds);
  }

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  if (loc_concat_) {
    const Dtype *loc_data = bottom[0]->cpu_data();
    GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                      share_location_, &all_loc_preds);
  } else {
    all_loc_preds.clear();
    all_loc_preds.resize(num);
    if (share_location_) {
      CHECK_EQ(num_loc_classes_, 1);
    }

    for (int i = 0; i < num; ++i) {
      for (int n = 0; n < nbottom_; n++) {
        const Dtype *loc_data = bottom[n + nbottom_]->cpu_data();
        LabelBBox &label_bbox = all_loc_preds[i];
        if (!ratio_permute_ && !no_permute_) // original caffe ssd
        {
          for (int p = 0;
               p < bottom[n + nbottom_]->channels() / num_loc_classes_ / 4;
               ++p) {
            int start_idx = p * num_loc_classes_ * 4;
            for (int c = 0; c < num_loc_classes_; ++c) {
              int label = share_location_ ? -1 : c;
              // if (label_bbox.find(label) == label_bbox.end()) {
              //  label_bbox[label].resize(num_priors_);
              //}
              NormalizedBBox locbox;
              locbox.set_xmin(loc_data[start_idx + c * 4]);
              locbox.set_ymin(loc_data[start_idx + c * 4 + 1]);
              locbox.set_xmax(loc_data[start_idx + c * 4 + 2]);
              locbox.set_ymax(loc_data[start_idx + c * 4 + 3]);
              float locbox_size = BBoxSize(locbox);
              locbox.set_size(locbox_size);
              label_bbox[label].push_back(locbox);
              // LOG(INFO)<<"origin p="<<p<<" ,xmin="<<loc_data[start_idx + c *
              // 4]<<" ,ymin="<<loc_data[start_idx + c * 4 + 1]<<
              //    " ,xmax="<<loc_data[start_idx + c * 4 + 2]<<"
              //    ,ymax="<<loc_data[start_idx + c * 4 + 3]<<"\n";
            }
          }
          // LOG(INFO)<<"loc num: "<<all_loc_preds[0][-1].size()<<"\n";
        } else if (ratio_permute_) {
          int count = bottom[n + nbottom_]->channels() / num_loc_classes_ / 4;
          for (int p = 0; p < count; ++p) {
            int start_idx = p * num_loc_classes_;
            for (int c = 0; c < num_loc_classes_; ++c) {
              int label = share_location_ ? -1 : c;
              // if (label_bbox.find(label) == label_bbox.end()) {
              //  label_bbox[label].resize(num_priors_);
              //}
              NormalizedBBox locbox;
              locbox.set_xmin(loc_data[start_idx + c * count]);
              locbox.set_ymin(loc_data[start_idx + c * count + 1 * count]);
              locbox.set_xmax(loc_data[start_idx + c * count + 2 * count]);
              locbox.set_ymax(loc_data[start_idx + c * count + 3 * count]);
              float locbox_size = BBoxSize(locbox);
              locbox.set_size(locbox_size);
              label_bbox[label].push_back(locbox);
              // LOG(INFO)<<"ratio p="<<p<<" ,xmin="<<loc_data[start_idx + c *
              // count]<<" ,ymin="<<loc_data[start_idx + c * count + 1 *
              // count]<<
              //    " ,xmax="<<loc_data[start_idx + c * count + 2 * count]<<"
              //    ,ymax="<<loc_data[start_idx + c * count + 3 * count]<<"\n";
            }
          }
        } else if (no_permute_) // [ratio][4][y][x] -> [ratio][y][x] boxes
        {
          int count =
              bottom[n + nbottom_]->height() * bottom[n + nbottom_]->width();
          for (int r = 0;
               r < bottom[n + nbottom_]->channels() / 4 / num_loc_classes_;
               ++r) {
            int start_idx = r * num_loc_classes_ * 4 * count;
            for (int p = 0; p < count; ++p) {
              for (int c = 0; c < num_loc_classes_; ++c) {
                int label = share_location_ ? -1 : c;
                // if (label_bbox.find(label) == label_bbox.end()) {
                //  label_bbox[label].resize(num_priors_);
                //}
                NormalizedBBox locbox;
                locbox.set_xmin(loc_data[start_idx + c * 4 * count + p]);
                locbox.set_ymin(
                    loc_data[start_idx + c * 4 * count + p + count]);
                locbox.set_xmax(
                    loc_data[start_idx + c * 4 * count + p + 2 * count]);
                locbox.set_ymax(
                    loc_data[start_idx + c * 4 * count + p + 3 * count]);
                float locbox_size = BBoxSize(locbox);
                locbox.set_size(locbox_size);
                label_bbox[label].push_back(locbox);
                // LOG(INFO)<<"p="<<p<<", r="<<r<<" ,xmin="<<loc_data[start_idx
                // + c * 4 * count + p]<<
                //    " ,ymin="<<loc_data[start_idx + c * 4 * count + p + 1 *
                //    count]<<
                //    " ,xmax="<<loc_data[start_idx + c * 4 * count + p + 2 *
                //    count]<<
                //    " ,ymax="<<loc_data[start_idx + c * 4 * count + p + 3 *
                //    count]<<"\n";
              }
            }
          }
        }
      }
    }
  }

  // Retrieve all confidences.
  vector<map<int, vector<float>>> all_conf_scores;
  if (conf_concat_ && arm_conf_data != NULL) {
    const Dtype *conf_data = bottom[1]->cpu_data();
    OSGetConfidenceScores(conf_data, arm_conf_data, num, num_priors_,
                          num_classes_, &all_conf_scores, objectness_score_);
  } else {
    if (conf_concat_) {
      const Dtype *conf_data = bottom[1]->cpu_data();
      GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                          &all_conf_scores);
    } else {
      all_conf_scores.clear();
      all_conf_scores.resize(num);
      for (int i = 0; i < num; ++i) {
        for (int n = 0; n < nbottom_; n++) {
          const Dtype *conf_data = bottom[n]->cpu_data();
          map<int, vector<float>> &label_scores = all_conf_scores[i];
          if (!ratio_permute_ && !no_permute_) // original caffe ssd
          {
            for (int p = 0; p < bottom[n]->channels() / num_classes_; ++p) {
              int start_idx = p * num_classes_;
              for (int c = 0; c < num_classes_; ++c) {
                label_scores[c].push_back(conf_data[start_idx + c]);
                // LOG(INFO)<<"origin p="<<p<<" ,c="<<c<<"
                // ,score="<<conf_data[start_idx + c]<<"\n";
              }
            }
            // LOG(INFO)<<"conf num: "<<all_conf_scores[0][0].size()<<"\n";
          } else if (ratio_permute_) {
            int count = bottom[n]->channels() / num_classes_;
            for (int p = 0; p < count; ++p) {
              int start_idx = p;
              for (int c = 0; c < num_classes_; ++c) {
                label_scores[c].push_back(conf_data[start_idx + c * count]);
                // LOG(INFO)<<"ratio p="<<p<<" ,c="<<c<<"
                // ,score="<<conf_data[start_idx + c * count]<<"\n";
              }
            }
          } else if (no_permute_) // [class][ratio][y][x]
          {
            for (int c = 0; c < num_classes_; ++c) {
              int start_idx = c * bottom[n]->height() * bottom[n]->width();
              for (int r = 0; r < bottom[n]->height(); ++r) {
                for (int p = 0; p < bottom[n]->width(); ++p) {
                  label_scores[c].push_back(
                      conf_data[start_idx + r * bottom[n]->width() + p]);
                  // LOG(INFO)<<"p="<<p<<" ,c="<<c<<"
                  // ,score="<<conf_data[start_idx + r * bottom[n]->width() +
                  // p]<<"\n";
                }
              }
            }
          }
        }
      }
    }
  }

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float>> prior_variances;
  if (priorbox_concat_) {
    if (!conf_concat_ && !loc_concat_) {
      const Dtype *prior_data = bottom[2 * nbottom_]->cpu_data();
      if (!ratio_permute_ && !no_permute_) // original caffe ssd
        GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes,
                       &prior_variances);
      else if (ratio_permute_) {
        prior_bboxes.clear();
        prior_variances.clear();
        int ratios[6] = {ratio0_, ratio1_, ratio2_,
                         ratio3_, ratio4_, ratio5_}; //
        int sum = 0;
        for (int n = 0; n < nbottom_; n++) {
          int count = bottom[n + nbottom_]->channels() /
                      4; // use loc shape to separate the priorbox data
          for (int i = sum; i < count + sum; ++i) {
            // "merge" the permute operation by index transform
            int xy_num = count / ratios[n];
            int permute_index =
                (i - sum) % xy_num * ratios[n] + (i - sum) / xy_num;
            int start_idx = (permute_index + sum) * 4; // i * 4;
            NormalizedBBox bbox;
            bbox.set_xmin(prior_data[start_idx]);
            bbox.set_ymin(prior_data[start_idx + 1]);
            bbox.set_xmax(prior_data[start_idx + 2]);
            bbox.set_ymax(prior_data[start_idx + 3]);
            float bbox_size = BBoxSize(bbox);
            bbox.set_size(bbox_size);
            prior_bboxes.push_back(bbox);
            // LOG(INFO)<<"i="<<i<<" ,xmin="<<prior_data[start_idx]<<"
            // ,ymin="<<prior_data[start_idx + 1]<<
            //    " ,xmax="<<prior_data[start_idx + 2]<<"
            //    ,ymax="<<prior_data[start_idx + 3]<<"\n";
          }

          for (int i = sum; i < count + sum; ++i) {
            int start_idx = (num_priors_ + i) * 4;
            vector<float> var;
            for (int j = 0; j < 4; ++j) {
              var.push_back(prior_data[start_idx + j]);
              // LOG(INFO)<<prior_data[start_idx + j]<<" ";
            }
            prior_variances.push_back(var);
          }
          sum += count;
        }
        // for priorbox with extra pre-permute layer cases
        /*
        int sum = 0;
        for(int n=0;n<6;n++)
        {
          int count = bottom[n+6]->channels()/4; // use loc shape to separate
        the priorbox data
          for (int i = sum; i < count+sum; ++i) {
            int start_idx = i;
            NormalizedBBox bbox;
            bbox.set_xmin(prior_data[start_idx]);
            bbox.set_ymin(prior_data[start_idx + 1 * count]);
            bbox.set_xmax(prior_data[start_idx + 2 * count]);
            bbox.set_ymax(prior_data[start_idx + 3 * count]);
            float bbox_size = BBoxSize(bbox);
            bbox.set_size(bbox_size);
            prior_bboxes.push_back(bbox);
            //LOG(INFO)<<"i="<<i<<" ,xmin="<<prior_data[start_idx]<<"
        ,ymin="<<prior_data[start_idx + 1 * count]<<
            //    " ,xmax="<<prior_data[start_idx + 2 * count]<<"
        ,ymax="<<prior_data[start_idx + 3 * count]<<"\n";
          }

          for (int i = sum; i < count+sum; ++i) {
            int start_idx = num_priors_ * 4 + i;
            vector<float> var;
            for (int j = 0; j < 4; ++j) {
              var.push_back(prior_data[start_idx + j * count]);
              //OG(INFO)<<prior_data[start_idx + j * count]<<" ";
            }
            prior_variances.push_back(var);
          }
          sum += count*4;
        }
        */
      } else if (no_permute_) // [y][x][ratio][4] -> [ratio][y][x] boxes
      {
        prior_bboxes.clear();
        prior_variances.clear();
        int sum = 0;
        for (int n = 0; n < nbottom_; n++) {
          int count =
              bottom[n + nbottom_]->channels() / 4 *
              bottom[n + nbottom_]->height() *
              bottom[n + nbottom_]
                  ->width(); // use loc shape to separate the priorbox data
          for (int i = sum; i < count + sum; ++i) {
            // "merge" the permute operation by index transform
            int xy_num = count / collect_ratios_[n];
            int permute_index =
                (i - sum) % xy_num * collect_ratios_[n] + (i - sum) / xy_num;
            int start_idx = (permute_index + sum) * 4;
            NormalizedBBox bbox;
            bbox.set_xmin(prior_data[start_idx]);
            bbox.set_ymin(prior_data[start_idx + 1]);
            bbox.set_xmax(prior_data[start_idx + 2]);
            bbox.set_ymax(prior_data[start_idx + 3]);
            float bbox_size = BBoxSize(bbox);
            bbox.set_size(bbox_size);
            prior_bboxes.push_back(bbox);
            // LOG(INFO)<<"i="<<i<<" ,xmin="<<prior_data[start_idx]<<"
            // ,ymin="<<prior_data[start_idx + 1]<<
            //    " ,xmax="<<prior_data[start_idx + 2]<<"
            //    ,ymax="<<prior_data[start_idx + 3]<<"\n";
          }

          for (int i = sum; i < count + sum; ++i) {
            int start_idx = (num_priors_ + i) * 4;
            vector<float> var;
            for (int j = 0; j < 4; ++j) {
              var.push_back(prior_data[start_idx + j]);
              // LOG(INFO)<<prior_data[start_idx + j]<<" ";
            }
            prior_variances.push_back(var);
          }
          sum += count;
        }
      }
    } else {
      const Dtype *prior_data = bottom[2]->cpu_data();
      if (!tflite_detection_) {
        GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes,
                       &prior_variances);
      } else {
        GetTFLiteBBoxes(prior_data, num_priors_, &prior_bboxes);
      }
    }
  } else {
    prior_bboxes.clear();
    prior_variances.clear();
    for (int n = 0; n < nbottom_; n++) {
      const Dtype *prior_data = bottom[n + 2 * nbottom_]->cpu_data();
      if (!ratio_permute_ && !no_permute_) // original caffe ssd
      {
        for (int i = 0; i < bottom[n + 2 * nbottom_]->height() / 4; ++i) {
          int start_idx = i * 4;
          NormalizedBBox bbox;
          bbox.set_xmin(prior_data[start_idx]);
          bbox.set_ymin(prior_data[start_idx + 1]);
          bbox.set_xmax(prior_data[start_idx + 2]);
          bbox.set_ymax(prior_data[start_idx + 3]);
          float bbox_size = BBoxSize(bbox);
          bbox.set_size(bbox_size);
          prior_bboxes.push_back(bbox);
          // LOG(INFO)<<"origin i="<<i<<" ,xmin="<<prior_data[start_idx]<<"
          // ,ymin="<<prior_data[start_idx + 1]<<
          //    " ,xmax="<<prior_data[start_idx + 2]<<"
          //    ,ymax="<<prior_data[start_idx + 3]<<"\n";
        }

        for (int i = 0; i < bottom[n + 2 * nbottom_]->height() / 4; ++i) {
          int start_idx = (bottom[n + 2 * nbottom_]->height() / 4 + i) * 4;
          vector<float> var;
          for (int j = 0; j < 4; ++j) {
            var.push_back(prior_data[start_idx + j]);
            // LOG(INFO)<<prior_data[start_idx + j]<<" ";
          }
          prior_variances.push_back(var);
        }
        // LOG(INFO)<<"prior num: "<<prior_bboxes.size()<<"\n";
      } else if (ratio_permute_) {
        int count = bottom[n + 2 * nbottom_]->height() / 4;
        for (int i = 0; i < count; ++i) {
          int start_idx = i;
          NormalizedBBox bbox;
          bbox.set_xmin(prior_data[start_idx]);
          bbox.set_ymin(prior_data[start_idx + 1 * count]);
          bbox.set_xmax(prior_data[start_idx + 2 * count]);
          bbox.set_ymax(prior_data[start_idx + 3 * count]);
          float bbox_size = BBoxSize(bbox);
          bbox.set_size(bbox_size);
          prior_bboxes.push_back(bbox);
          // LOG(INFO)<<"index i="<<i<<" ,xmin="<<prior_data[start_idx]<<"
          // ,ymin="<<prior_data[start_idx + 1 * count]<<
          //    " ,xmax="<<prior_data[start_idx + 2 * count]<<"
          //    ,ymax="<<prior_data[start_idx + 3 * count]<<"\n";
        }

        for (int i = 0; i < count; ++i) {
          int start_idx = count * 4 + i;
          vector<float> var;
          for (int j = 0; j < 4; ++j) {
            var.push_back(prior_data[start_idx + j * count]);
            // LOG(INFO)<<prior_data[start_idx + j * count]<<" ";
          }
          prior_variances.push_back(var);
        }
      }
    }
  }

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  const bool clip_bbox = false;
  if (bottom.size() >= 5 && loc_concat_) {
    CasRegDecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                          share_location_, num_loc_classes_,
                          background_label_id_, code_type_,
                          variance_encoded_in_target_, clip_bbox,
                          &all_decode_bboxes, all_arm_loc_preds);
  } else {
    if (!tflite_detection_) {
      DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                      share_location_, num_loc_classes_, background_label_id_,
                      code_type_, variance_encoded_in_target_, clip_bbox,
                      &all_decode_bboxes);
    } else {
      DecodeBBoxesTFLite(all_loc_preds, prior_bboxes, num, scale_xywh_,
                         &all_decode_bboxes);
    }
  }

  int num_kept = 0;
  if (!tflite_detection_) {
    vector<map<int, vector<int>>> all_indices;
    for (int i = 0; i < num; ++i) {
      const LabelBBox &decode_bboxes = all_decode_bboxes[i];
      const map<int, vector<float>> &conf_scores = all_conf_scores[i];
      map<int, vector<int>> indices;
      int num_det = 0;
      for (int c = 0; c < num_classes_; ++c) {
        if (c == background_label_id_) {
          // Ignore background class.
          continue;
        }
        if (conf_scores.find(c) == conf_scores.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          LOG(FATAL) << "Could not find confidence predictions for label " << c;
        }
        const vector<float> &scores = conf_scores.find(c)->second;
        int label = share_location_ ? -1 : c;
        if (decode_bboxes.find(label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          LOG(FATAL) << "Could not find location predictions for label "
                     << label;
          continue;
        }
        const vector<NormalizedBBox> &bboxes =
            decode_bboxes.find(label)->second;
        ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
                     eta_, top_k_, &(indices[c]));
        num_det += indices[c].size();
      }
      if (keep_top_k_ > -1 && num_det > keep_top_k_) {
        vector<pair<float, pair<int, int>>> score_index_pairs;
        for (map<int, vector<int>>::iterator it = indices.begin();
             it != indices.end(); ++it) {
          int label = it->first;
          const vector<int> &label_indices = it->second;
          if (conf_scores.find(label) == conf_scores.end()) {
            // Something bad happened for current label.
            LOG(FATAL) << "Could not find location predictions for " << label;
            continue;
          }
          const vector<float> &scores = conf_scores.find(label)->second;
          for (int j = 0; j < label_indices.size(); ++j) {
            int idx = label_indices[j];
            CHECK_LT(idx, scores.size());
            score_index_pairs.push_back(
                std::make_pair(scores[idx], std::make_pair(label, idx)));
          }
        }
        // Keep top k results per image.
        std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                  SortScorePairDescend<pair<int, int>>);
        score_index_pairs.resize(keep_top_k_);
        // Store the new indices.
        map<int, vector<int>> new_indices;
        for (int j = 0; j < score_index_pairs.size(); ++j) {
          int label = score_index_pairs[j].second.first;
          int idx = score_index_pairs[j].second.second;
          new_indices[label].push_back(idx);
        }
        all_indices.push_back(new_indices);
        num_kept += keep_top_k_;
      } else {
        all_indices.push_back(indices);
        num_kept += num_det;
      }
    }

    vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(7);
    Dtype *top_data;
    if (num_kept == 0) {
      LOG(INFO) << "Couldn't find any detections";
      top_shape[2] = num;
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
      caffe_set<Dtype>(top[0]->count(), -1, top_data);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 7;
      }
    } else {
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
    }

    int count = 0;
    boost::filesystem::path output_directory(output_directory_);
    for (int i = 0; i < num; ++i) {
      const map<int, vector<float>> &conf_scores = all_conf_scores[i];
      const LabelBBox &decode_bboxes = all_decode_bboxes[i];
      for (map<int, vector<int>>::iterator it = all_indices[i].begin();
           it != all_indices[i].end(); ++it) {
        int label = it->first;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          LOG(FATAL) << "Could not find confidence predictions for " << label;
          continue;
        }
        const vector<float> &scores = conf_scores.find(label)->second;
        int loc_label = share_location_ ? -1 : label;
        if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          LOG(FATAL) << "Could not find location predictions for " << loc_label;
          continue;
        }
        const vector<NormalizedBBox> &bboxes =
            decode_bboxes.find(loc_label)->second;
        vector<int> &indices = it->second;
        if (need_save_) {
          CHECK(label_to_name_.find(label) != label_to_name_.end())
              << "Cannot find label: " << label << " in the label map.";
          CHECK_LT(name_count_, names_.size());
        }
        for (int j = 0; j < indices.size(); ++j) {
          int idx = indices[j];
          top_data[count * 7] = i;
          top_data[count * 7 + 1] = label;
          top_data[count * 7 + 2] = scores[idx];
          const NormalizedBBox &bbox = bboxes[idx];
          top_data[count * 7 + 3] = bbox.xmin();
          top_data[count * 7 + 4] = bbox.ymin();
          top_data[count * 7 + 5] = bbox.xmax();
          top_data[count * 7 + 6] = bbox.ymax();
          if (need_save_) {
            NormalizedBBox out_bbox;
            OutputBBox(bbox, sizes_[name_count_], has_resize_, resize_param_,
                       &out_bbox);
            float score = top_data[count * 7 + 2];
            float xmin = out_bbox.xmin();
            float ymin = out_bbox.ymin();
            float xmax = out_bbox.xmax();
            float ymax = out_bbox.ymax();
            ptree pt_xmin, pt_ymin, pt_width, pt_height;
            pt_xmin.put<float>("", round(xmin * 100) / 100.);
            pt_ymin.put<float>("", round(ymin * 100) / 100.);
            pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
            pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

            ptree cur_bbox;
            cur_bbox.push_back(std::make_pair("", pt_xmin));
            cur_bbox.push_back(std::make_pair("", pt_ymin));
            cur_bbox.push_back(std::make_pair("", pt_width));
            cur_bbox.push_back(std::make_pair("", pt_height));

            ptree cur_det;
            cur_det.put("image_id", names_[name_count_]);
            if (output_format_ == "ILSVRC") {
              cur_det.put<int>("category_id", label);
            } else {
              cur_det.put("category_id", label_to_name_[label].c_str());
            }
            cur_det.add_child("bbox", cur_bbox);
            cur_det.put<float>("score", score);

            detections_.push_back(std::make_pair("", cur_det));
          }
          ++count;
        }
      }
      if (need_save_) {
        ++name_count_;
        if (name_count_ % num_test_image_ == 0) {
          if (output_format_ == "VOC") {
            map<string, std::ofstream *> outfiles;
            for (int c = 0; c < num_classes_; ++c) {
              if (c == background_label_id_) {
                continue;
              }
              string label_name = label_to_name_[c];
              boost::filesystem::path file(output_name_prefix_ + label_name +
                                           ".txt");
              boost::filesystem::path out_file = output_directory / file;
              outfiles[label_name] = new std::ofstream(
                  out_file.string().c_str(), std::ofstream::out);
            }
            BOOST_FOREACH (ptree::value_type &det, detections_.get_child("")) {
              ptree pt = det.second;
              string label_name = pt.get<string>("category_id");
              if (outfiles.find(label_name) == outfiles.end()) {
                std::cout << "Cannot find " << label_name << std::endl;
                continue;
              }
              string image_name = pt.get<string>("image_id");
              float score = pt.get<float>("score");
              vector<float> bbox;
              BOOST_FOREACH (ptree::value_type &elem, pt.get_child("bbox")) {
                bbox.push_back(
                    static_cast<float>(elem.second.get_value<float>()));
              }
              *(outfiles[label_name]) << image_name;
              *(outfiles[label_name]) << " " << score;
              *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
              *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
              *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
              *(outfiles[label_name]) << std::endl;
            }
            for (int c = 0; c < num_classes_; ++c) {
              if (c == background_label_id_) {
                continue;
              }
              string label_name = label_to_name_[c];
              outfiles[label_name]->flush();
              outfiles[label_name]->close();
              delete outfiles[label_name];
            }
          } else if (output_format_ == "COCO") {
            boost::filesystem::path output_directory(output_directory_);
            boost::filesystem::path file(output_name_prefix_ + ".json");
            boost::filesystem::path out_file = output_directory / file;
            std::ofstream outfile;
            outfile.open(out_file.string().c_str(), std::ofstream::out);

            boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
            ptree output;
            output.add_child("detections", detections_);
            std::stringstream ss;
            write_json(ss, output);
            std::string rv = boost::regex_replace(ss.str(), exp, "$1");
            outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
                    << std::endl
                    << "]" << std::endl;
          } else if (output_format_ == "ILSVRC") {
            boost::filesystem::path output_directory(output_directory_);
            boost::filesystem::path file(output_name_prefix_ + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            std::ofstream outfile;
            outfile.open(out_file.string().c_str(), std::ofstream::out);

            BOOST_FOREACH (ptree::value_type &det, detections_.get_child("")) {
              ptree pt = det.second;
              int label = pt.get<int>("category_id");
              string image_name = pt.get<string>("image_id");
              float score = pt.get<float>("score");
              vector<int> bbox;
              BOOST_FOREACH (ptree::value_type &elem, pt.get_child("bbox")) {
                bbox.push_back(
                    static_cast<int>(elem.second.get_value<float>()));
              }
              outfile << image_name << " " << label << " " << score;
              outfile << " " << bbox[0] << " " << bbox[1];
              outfile << " " << bbox[0] + bbox[2];
              outfile << " " << bbox[1] + bbox[3];
              outfile << std::endl;
            }
          }
          name_count_ = 0;
          detections_.clear();
        }
      }
    }
    if (visualize_) {
#ifdef USE_OPENCV
      vector<cv::Mat> cv_imgs;
      this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
      vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
      VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
                    label_to_display_name_, save_file_);
#endif // USE_OPENCV
    }
  } else {
    const int num_categories_per_anchor =
        std::min<Dtype>(max_classes_per_detection_, num_classes_);
    int count = 0;
    const Dtype *conf_data = bottom[1]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < num; ++i) {
      const LabelBBox &decode_bboxes = all_decode_bboxes[i];
      map<int, vector<int>> indices;
      // tflite process
      vector<Dtype> max_scores(num_priors_);
      vector<vector<int>> sorted_class_indices(num_priors_);

      // get the max score in each row
      for (int row = 0; row < num_priors_; row++) {
        if (background_label_id_ == 0) {
          vector<float> box_scores(num_classes_ - 1);
          vector<int> class_indices(num_classes_ - 1);
          for (int c = 1; c < num_classes_; c++) {
            box_scores[c - 1] = conf_data[i * num_priors_ * num_classes_ +
                                          row * num_classes_ + c];
            class_indices[c - 1] = c - 1;
          }
          std::partial_sort(class_indices.begin(),
                            class_indices.begin() + num_categories_per_anchor,
                            class_indices.end(), [&](int i, int j) {
                              return box_scores[i] > box_scores[j];
                            });
          max_scores[row] = box_scores[class_indices[0]];
          sorted_class_indices[row] = class_indices;
        }
      }

      // Generate index score pairs.
      vector<pair<float, int>> score_index_vec;
      score_index_vec.clear();
      for (int i = 0; i < num_priors_; ++i) {
        if (max_scores[i] >= confidence_threshold_) {
          score_index_vec.push_back(std::make_pair(max_scores[i], i));
        }
      }

      // Sort the score pair according to the scores in descending order
      std::sort(score_index_vec.begin(), score_index_vec.end(),
                SortScorePairDescend<int>);

      int num_scores_kept = score_index_vec.size();
      const int output_size = std::min<int>(keep_top_k_, num_scores_kept);

      int label = -1;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;

      // nms part: for all boxes satisfying scores > score_threshold
      vector<int> select_indices;
      select_indices.clear();

      int num_active_candidate = num_scores_kept;
      vector<int> active(num_scores_kept, 1);
      for (int m = 0; m < num_scores_kept; m++) {
        if ((num_active_candidate == 0) ||
            (select_indices.size() >= output_size)) {
          break;
        }
        int idx = score_index_vec[m].second;
        if (active[m] == 1) {
          select_indices.push_back(idx);
          active[m] = 0;
          num_active_candidate -= 1;
        } else {
          continue;
        }
        for (int n = m + 1; n < num_scores_kept; n++) {
          if (active[n] == 1) {
            int n_idx = score_index_vec[n].second;
            float overlap = JaccardOverlap(bboxes[idx], bboxes[n_idx]);
            if (overlap > nms_threshold_) {
              active[n] = 0;
              num_active_candidate -= 1;
            }
          }
        }
      }

      num_kept += select_indices.size() * num_categories_per_anchor;
      // select based on max scores in each row

      for (int j = 0; j < select_indices.size(); j++) {
        for (int col = 0; col < num_categories_per_anchor; col++) {
          int d_class = sorted_class_indices[select_indices[j]][col];
          float d_score =
              conf_data[select_indices[j] * num_classes_ + 1 + d_class];

          top_data[count * 7] = i;
          top_data[count * 7 + 1] = d_class;
          top_data[count * 7 + 2] = d_score;
          const NormalizedBBox &bbox = bboxes[select_indices[j]];
          top_data[count * 7 + 3] = bbox.xmin();
          top_data[count * 7 + 4] = bbox.ymin();
          top_data[count * 7 + 5] = bbox.xmax();
          top_data[count * 7 + 6] = bbox.ymax();

          ++count;
        }
      }
    }

    vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(7);
    if (num_kept == 0) {
      LOG(INFO) << "Couldn't find any detections";
      top_shape[2] = num;
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
      caffe_set<Dtype>(top[0]->count(), -1, top_data);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 7;
      }
    } else {
      top[0]->Reshape(top_shape);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectionOutputLayer);
REGISTER_LAYER_CLASS(DetectionOutput);

} // namespace caffe
