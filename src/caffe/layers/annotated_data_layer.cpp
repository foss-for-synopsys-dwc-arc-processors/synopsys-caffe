#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    //reader_(param) {
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DataParameter param = this->layer_param_.data_param();
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  //AnnotatedDatum& anno_datum = *(reader_.full().peek());
  AnnotatedDatum anno_datum;
  anno_datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    if(transform_param.has_caffe_yolo()) {
      this->box_label_ = true;
      vector<int> label_shape(1, batch_size);
      if (param.side_size() > 0) {
        for (int i = 0; i < param.side_size(); ++i) {
          sides_.push_back(param.side(i));
        }
      }
      CHECK_EQ(sides_.size(), top.size() - 1) << "side num not equal to top size";
      if (has_anno_type_) {
        anno_type_ = anno_datum.type();
        if (anno_data_param.has_anno_type()) {
          // If anno_type is provided in AnnotatedDataParameter, replace
          // the type stored in each individual AnnotatedDatum.
          LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
          anno_type_ = anno_data_param.anno_type();
        }
        for (int i = 0; i < this->prefetch_.size(); ++i) {
          this->prefetch_[i]->multi_label_.clear();
        }
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Yolo label format
          for (int i = 0; i < sides_.size(); ++i) {
            vector<int> label_shape(1, batch_size);
            int label_size = sides_[i] * sides_[i] * (1 + 1 + 1 + 4);
            label_shape.push_back(label_size);
            top[i+1]->Reshape(label_shape);
            for (int j = 0; j < this->prefetch_.size(); ++j) {
              shared_ptr<Blob<Dtype> > tmp_blob;
              tmp_blob.reset(new Blob<Dtype>(label_shape));
              this->prefetch_[j]->multi_label_.push_back(tmp_blob);
            }
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
      } else {
        label_shape[0] = batch_size;
      }
    }
    else {
      vector<int> label_shape(4, 1);
      if (has_anno_type_) {
        anno_type_ = anno_datum.type();
        if (anno_data_param.has_anno_type()) {
          // If anno_type is provided in AnnotatedDataParameter, replace
          // the type stored in each individual AnnotatedDatum.
          LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
          anno_type_ = anno_data_param.anno_type();
        }
        // Infer the label shape from anno_datum.AnnotationGroup().
        int num_bboxes = 0;
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Since the number of bboxes can be different for each image,
          // we store the bbox information in a specific format. In specific:
          // All bboxes are stored in one spatial plane (num and channels are 1)
          // And each row contains one and only one box in the following format:
          // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
          // Note: Refer to caffe.proto for details about group_label and
          // instance_id.
          for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
            num_bboxes += anno_datum.annotation_group(g).annotation_size();
          }
          label_shape[0] = 1;
          label_shape[1] = 1;
          // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
          // cpu_data and gpu_data for consistent prefetch thread. Thus we make
          // sure there is at least one bbox.
          label_shape[2] = std::max(num_bboxes, 1);
          label_shape[3] = 8;
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
      } else {
        label_shape[0] = batch_size;
      }
      top[1]->Reshape(label_shape);
      for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
      }
    }
  }
}

template <typename Dtype>
bool AnnotatedDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void AnnotatedDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();

  AnnotatedDatum anno_datum;
  //AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  //vector<int> top_shape =
  //    this->data_transformer_->InferBlobShape(anno_datum.datum());
  //this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  //top_shape[0] = batch_size;
  //batch->data_.Reshape(top_shape);

  //Dtype* top_data = batch->data_.mutable_cpu_data();
  //Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  //if (this->output_labels_ && !has_anno_type_) {
  //  top_label = batch->label_.mutable_cpu_data();
  //}

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;
  vector<Dtype*> top_label;
  for (int i = 0; i < sides_.size(); ++i) {
    top_label.push_back(batch->multi_label_[i]->mutable_cpu_data());
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    //AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    while (Skip()) {
      Next();
    }
    anno_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    //timer.Start();
    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> top_shape =
        this->data_transformer_->InferBlobShape(anno_datum.datum());
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        //top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
        if (transform_param.has_caffe_yolo() && (anno_type_ == AnnotatedDatum_AnnotationType_BBOX)) {
          vector<int> label_shape(2);
          int count = 0;
          int label_offset = 0;
          int side = 0;
          for (int i = 0; i < sides_.size(); ++i) {
            side = sides_[i];
            count = sides_[i] * sides_[i] * (1 + 1 + 1 + 4);
            label_shape[0] = batch_size;
            label_shape[1] = count;
            batch->multi_label_[i]->Reshape(label_shape);
            top_label[i] = batch->multi_label_[i]->mutable_cpu_data();
            const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
            label_offset = count * item_id;
            top_label[i] = top_label[i] + label_offset;
            int locations = pow(side, 2);
            CHECK_EQ(count, locations * 7) << "side and count not match";
            // difficult
            caffe_set(locations, Dtype(0), top_label[i]);
            // isobj
            caffe_set(locations, Dtype(0), top_label[i] + locations);
            // class label
            caffe_set(locations, Dtype(-1), top_label[i] + locations * 2);
            // bounding box
            caffe_set(locations*4, Dtype(0), top_label[i] + locations * 3);
            for (int g = 0; g < anno_vec.size(); ++g) {
              const AnnotationGroup& anno_group = anno_vec[g];
              for (int a = 0; a < anno_group.annotation_size(); ++a) {
                const Annotation& anno = anno_group.annotation(a);
                const NormalizedBBox& bbox = anno.bbox();
                float class_label = anno_group.group_label();
                float x = bbox.x_center();
                float y = bbox.y_center();
                int x_index = floor(x * side);
                int y_index = floor(y * side);
                x_index = std::min(x_index, side - 1);
                y_index = std::min(y_index, side - 1);
                int dif_index = side * y_index + x_index;
                int obj_index = locations + dif_index;
                int class_index = locations * 2 + dif_index;
                int cor_index = locations * 3 + dif_index * 4;
                top_label[i][dif_index] = bbox.difficult();
                top_label[i][obj_index] = 1;
                top_label[i][class_index] = class_label;
                top_label[i][cor_index + 0] = bbox.x_center();
                top_label[i][cor_index + 1] = bbox.y_center();
                top_label[i][cor_index + 2] = bbox.width();
                top_label[i][cor_index + 3] = bbox.height();
              }
            }
          }
        }
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        Dtype* top_label = batch->label_.mutable_cpu_data();
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    //reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
    Next();
  }
  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    if(anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      if (!(transform_param.has_caffe_yolo())) {
        vector<int> label_shape(4);
        label_shape[0] = 1;
        label_shape[1] = 1;
        label_shape[3] = 8;
        if (num_bboxes == 0) {
          // Store all -1 in the label.
          label_shape[2] = 1;
          batch->label_.Reshape(label_shape);
          caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
        } else {
          // Reshape the label and store the annotation.
          label_shape[2] = num_bboxes;
          batch->label_.Reshape(label_shape);
          Dtype* top_label = batch->label_.mutable_cpu_data();
          int idx = 0;
          for (int item_id = 0; item_id < batch_size; ++item_id) {
            const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
            for (int g = 0; g < anno_vec.size(); ++g) {
              const AnnotationGroup& anno_group = anno_vec[g];
              for (int a = 0; a < anno_group.annotation_size(); ++a) {
                const Annotation& anno = anno_group.annotation(a);
                const NormalizedBBox& bbox = anno.bbox();
                top_label[idx++] = item_id;
                top_label[idx++] = anno_group.group_label();
                top_label[idx++] = anno.instance_id();
                top_label[idx++] = bbox.xmin();
                top_label[idx++] = bbox.ymin();
                top_label[idx++] = bbox.xmax();
                top_label[idx++] = bbox.ymax();
                top_label[idx++] = bbox.difficult();
              }
            }
          }
        }
      }
    }
    else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
